"""m4aplay.py — the GBA's MP2K ("Sappy", m4a) sound driver, ported, logging
instead of mixing.

A port of pret's decompilation (pokeemerald/pmd-red `src/m4a_1.s`, `src/m4a.c`,
`src/m4a_tables.c`; verbatim in tsvm reference_materials/gba-m4a/): MPlayMain's
tick loop (tempo counter, gate times, the command table, the LFO), ply_note's
tone resolution (key splits, drum tables) and channel allocation (DirectSound
by priority and track order, preferring releasing channels; the four CGB
channels one per type), TrkVolPitSet / ChnVolSetAsm (volume, pan, pitch), the
mixer's DirectSound envelope (SoundMainRAM) and CgbSound's PSG envelope. The
envelopes are run only so channels free up when the driver frees them; the
.taud hears envelopes from the SoundFont (m4abank.py).

Positions are SEQUENCE TICKS (24 a quarter note); a frame is 1/59.7275 s.
Out: notes (Note), per-track curves (pitch, level, pan per frame), tempo log,
loop window.
"""

import math
import struct

from gbarom import ROM_BASE, u32, is_ptr

FPS = 16777216.0 / 280896.0          # the LCD's 59.7275 Hz
SEQ_PPQN = 24

CLOCK = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 28, 30, 32, 36, 40, 42, 44, 48, 52, 54, 56, 60, 64, 66,
         68, 72, 76, 78, 80, 84, 88, 90, 92, 96]
PCM_PER_VBLANK = [96, 132, 176, 224, 264, 304, 352, 448, 528, 608, 672, 704]
CGB_SCALE = [((i // 12) << 4) | (i % 12) for i in range(132)]
CGB_FREQ = [-2004, -1891, -1785, -1685, -1591, -1501, -1417, -1337, -1262, -1192, -1125, -1062]
NOISE = [0xD7, 0xD6, 0xD5, 0xD4, 0xC7, 0xC6, 0xC5, 0xC4, 0xB7, 0xB6, 0xB5, 0xB4,
         0xA7, 0xA6, 0xA5, 0xA4, 0x97, 0x96, 0x95, 0x94, 0x87, 0x86, 0x85, 0x84,
         0x77, 0x76, 0x75, 0x74, 0x67, 0x66, 0x65, 0x64, 0x57, 0x56, 0x55, 0x54,
         0x47, 0x46, 0x45, 0x44, 0x37, 0x36, 0x35, 0x34, 0x27, 0x26, 0x25, 0x24,
         0x17, 0x16, 0x15, 0x14, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00]
CGB3_VOL = [0x00, 0x00, 0x60, 0x60, 0x60, 0x60, 0x40, 0x40, 0x40, 0x40,
            0x80, 0x80, 0x80, 0x80, 0x20, 0x20]

T_CGB, T_FIX, T_REV, T_CMP, T_SPL, T_RHY = 0x07, 0x08, 0x10, 0x20, 0x40, 0x80
SF_START, SF_STOP, SF_LOOP, SF_IEC, SF_ENV = 0x80, 0x40, 0x10, 0x04, 0x03
SF_ON = SF_START | SF_STOP | SF_IEC | SF_ENV
F_VOLSET, F_VOLCHG, F_PITSET, F_PITCHG, F_START, F_EXIST = 0x01, 0x03, 0x04, 0x0C, 0x40, 0x80


def s8(v):
    v &= 0xFF
    return v - 256 if v & 0x80 else v


def cgb_reg(cgb, key, fine=0):
    """MidiKeyToCgbFreq: the register value (square/wave 11-bit + 2048 marker;
    noise the NR43 byte)."""
    if cgb == 4:
        key = 0 if key <= 20 else min(59, key - 21)
        return NOISE[key]
    if key <= 35:
        fine, key = 0, 0
    else:
        key -= 36
        if key > 130:
            key, fine = 130, 255
    v1 = CGB_FREQ[CGB_SCALE[key] & 15] >> (CGB_SCALE[key] >> 4)
    v2 = CGB_FREQ[CGB_SCALE[key + 1] & 15] >> (CGB_SCALE[key + 1] >> 4)
    return v1 + ((fine * (v2 - v1)) >> 8) + 2048


def cgb_hz(cgb, reg):
    """The tone's fundamental (square, wave) or the LFSR clock (noise)."""
    if cgb == 4:
        s, r = reg >> 4, reg & 7
        return 524288.0 / (r if r else 0.5) / 2 ** (s + 1)
    x = reg & 0x7FF
    if x >= 2048 or x <= 0 and reg < 2048:
        x = max(0, min(2047, x))
    return (131072.0 if cgb in (1, 2) else 65536.0) / (2048 - x)


class Tone:
    __slots__ = ('type', 'key', 'length', 'ps', 'wav', 'a', 'd', 's', 'r', 'addr')

    @classmethod
    def read(cls, rom, addr):
        t = cls()
        o = addr - ROM_BASE
        b = rom[o:o + 12] if 0 <= o <= len(rom) - 12 else bytes(12)
        t.type, t.key, t.length, t.ps = b[0], b[1], b[2], b[3]
        t.wav = struct.unpack_from('<I', b, 4)[0]
        t.a, t.d, t.s, t.r = b[8], b[9], b[10], b[11]
        t.addr = addr
        return t

    def copy(self):
        t = Tone()
        for k in Tone.__slots__:
            setattr(t, k, getattr(self, k))
        return t

    def ident(self):
        """What a note played through this tone depends on (overrides included)."""
        return (self.type, self.key, self.length, self.ps, self.wav, self.a, self.d, self.s, self.r)


class Note:
    __slots__ = ('track', 'key', 'vel', 'tone', 'sub', 'cgb', 'goal', 'hwpan', 'start', 'end',
                 'why', 'serial', 'drum', 'kshift', 'rpan', 'echo', 'silent', 'cut')

    def __repr__(self):
        return (f'Note(t{self.track} k{self.key} v{self.vel} cgb{self.cgb} {self.start:.2f}–{self.end} '
                f'{self.why})')


class Chan:
    """A SoundChannel or a CgbChannel (the fields both use, and the CGB's own)."""

    def __init__(self, cgb=0):
        self.cgb = cgb                 # 0 = DirectSound, 1…4 = the CGB channel's type
        self.status = 0
        self.type = cgb
        self.right = self.left = 0
        self.a = self.d = self.s = self.r = 0
        self.key = self.midi_key = self.vel = self.prio = 0
        self.rpan = 0
        self.gate = 0
        self.env = 0
        self.echo_vol = self.echo_len = 0
        self.wav = 0
        self.track = None
        self.note = None
        self.last = None
        # DirectSound
        self.count = 0.0
        self.freq = 0.0
        self.wave = None
        # CGB
        self.goal = self.counter = self.sus_goal = 0
        self.pan = 0
        self.pan_mask = (0x11, 0x22, 0x44, 0x88)[cgb - 1] if cgb else 0
        self.length = 0
        self.sweep = 0
        self.len_frames = None


class Track:
    def __init__(self, idx, ptr):
        self.idx = idx
        self.flags = F_EXIST | F_START
        self.ptr = ptr
        self.chans = []                # newest first, as the driver prepends
        self.stack = []
        self.visited = {}
        self.loops = []                # (target's first tick, jump tick, silent)
        self.note_ticks = []
        self.timer = 0
        self.clear()
        self.flags = F_EXIST | F_START

    def clear(self):
        """Clear64byte + the START defaults (MPlayMain)."""
        self.wait = 0
        self.rep_n = 0
        self.gate = 0
        self.key = 0
        self.vel = 0
        self.running = 0
        self.key_m = 0
        self.pit_m = 0
        self.key_shift = 0
        self.tune = 0
        self.bend = 0
        self.bend_range = 2
        self.vol_mr = self.vol_ml = 0
        self.vol = 0
        self.vol_x = 64
        self.pan = 0
        self.y = 0
        self.mod_m = 0
        self.mod = 0
        self.mod_t = 0
        self.lfo_speed = 22
        self.lfo_speed_c = 0
        self.lfo_delay = self.lfo_delay_c = 0
        self.prio = 0
        self.x = 0
        self.echo_vol = self.echo_len = 0
        self.tone = Tone()
        self.tone.type, self.tone.key, self.tone.length, self.tone.ps, self.tone.wav = 1, 0, 0, 0, 0
        self.tone.a = self.tone.d = self.tone.s = self.tone.r = 0
        self.tone.addr = 0
        self.timer = 0
        self.start_off = 0
        self.stack = []

    def idle(self):
        return bool(self.loops) and self.loops[-1][2]


class WaveInfo:
    __slots__ = ('addr', 'freq', 'loop', 'loop_start', 'size', 'compressed')


class Player:
    def __init__(self, rom, m4a, header, max_tracks, warn=print):
        self.rom = rom
        self.warn = warn
        self.warned = set()
        o = header - ROM_BASE
        n, _blocks, prio, reverb = rom[o], rom[o + 1], rom[o + 2], rom[o + 3]
        self.prio = prio
        self.reverb = reverb
        self.voices = u32(rom, o + 4)
        ntr = min(n, max_tracks, 16)
        self.tracks = [Track(i, u32(rom, o + 8 + 4 * i)) for i in range(ntr)]
        self.mode = m4a.mode
        self.max_chans = m4a.max_chans
        self.master_volume = m4a.master_volume
        self.pcm_per_vblank = PCM_PER_VBLANK[m4a.freq_index - 1]
        self.pcm_freq = (597275 * self.pcm_per_vblank + 5000) // 10000
        self.ds = [Chan(0) for _ in range(self.max_chans)]
        self.cgbs = [Chan(i) for i in (1, 2, 3, 4)]
        self.c15 = 0
        self.tempo_d = 150
        self.tempo_u = 0x100
        self.tempo_i = 150
        self.tempo_c = 0
        self.clock = 0
        self.frame = 0
        self.status = 0
        self.memacc = bytearray(16)
        self.notes = []
        self.serial = 0
        self.tempo_log = [(0, 150)]
        self.curves = {t.idx: [] for t in self.tracks}
        self._last_curve = {}
        self.cgb_curves = {}           # track → [(pos, goal, hwpan, cgb type)]
        self._last_cgb = {}
        self.waves = {}
        self.dropped = 0
        self.vol_args = {}             # track → ROM addresses of its VOL arguments (for --solo)

    def note_warn(self, key, msg):
        if key not in self.warned:
            self.warned.add(key)
            self.warn(msg)

    # ── helpers ────────────────────────────────────────────────────────────────
    def byte(self, addr):
        o = addr - ROM_BASE
        return self.rom[o] if 0 <= o < len(self.rom) else 0

    def word(self, addr):
        return u32(self.rom, addr - ROM_BASE)

    def wave(self, addr):
        w = self.waves.get(addr)
        if w is None:
            w = WaveInfo()
            o = addr - ROM_BASE
            w.addr = addr
            w.compressed = struct.unpack_from('<H', self.rom, o)[0] if o >= 0 else 0
            w.loop = bool(self.rom[o + 3] & 0xC0) if o >= 0 else False
            w.freq = u32(self.rom, o + 4) / 1024.0
            w.loop_start = u32(self.rom, o + 8)
            w.size = u32(self.rom, o + 12)
            self.waves[addr] = w
        return w

    def pos_frame(self):
        return max(0.0, self.clock - 1 + self.tempo_c / 150.0)

    def end_note(self, ch, pos, why):
        n = ch.note
        if n is not None and n.end is None:
            n.end = max(n.start, pos)
            n.why = why
        ch.note = None

    def silence(self, ch, pos, cut):
        """The channel's last note has stopped sounding: its release ended, or
        (cut) a new note took the channel over while it still sounded."""
        n = ch.last
        if n is not None and n.silent is None:
            n.silent = max(n.end if n.end is not None else n.start, pos)
            n.cut = cut
        ch.last = None

    # ── TrkVolPitSet / ChnVolSetAsm ────────────────────────────────────────────
    def trk_vol_pit_set(self, t):
        if t.flags & F_VOLSET:
            x = (t.vol * t.vol_x) >> 5
            if t.mod_t == 1:
                x = (x * (t.mod_m + 128)) >> 7
            y = 2 * t.pan
            if t.mod_t == 2:
                y += t.mod_m
            y = max(-128, min(127, y))
            t.y = y
            t.vol_mr = ((y + 128) * x) >> 8
            t.vol_ml = ((127 - y) * x) >> 8
        if t.flags & F_PITSET:
            bend = t.bend * t.bend_range
            x = (t.tune + bend) * 4 + (t.key_shift << 8) + (16 * t.mod_m if t.mod_t == 0 else 0)
            t.key_m = s8(x >> 8)            # stored as a byte, read back signed (ldrsb)
            t.pit_m = x & 0xFF
            t.x = (t.key_m << 8) + t.pit_m
        t.flags &= ~(F_PITSET | F_VOLSET) & 0xFF

    @staticmethod
    def chn_vol_set(ch, t):
        r = (ch.vel * (128 + ch.rpan) * t.vol_mr) >> 14
        ch.right = min(255, r)
        l = (ch.vel * (127 - ch.rpan) * t.vol_ml) >> 14
        ch.left = min(255, l)

    # ── the command loop ───────────────────────────────────────────────────────
    def read(self, t):
        v = self.byte(t.ptr)
        t.ptr += 1
        return v

    def track_stop(self, t):
        if not (t.flags & F_EXIST):
            return
        for ch in t.chans:
            if ch.status:
                if ch.cgb:
                    self.cgb_osc_off(ch)
                ch.status = 0
                self.end_note(ch, self.clock, 'track-stop')
                self.silence(ch, float(self.clock), True)
            ch.track = None
        t.chans = []

    def ply_fine(self, t):
        for ch in t.chans:
            if ch.status & SF_ON:
                ch.status |= SF_STOP
                self.end_note(ch, self.clock, 'fine')
            ch.track = None
        t.chans = []
        t.flags = 0

    def jump(self, t, target, kind):
        if kind == 'goto' and target in t.visited:
            j = self.clock
            s = t.visited[target]
            silent = not any(s <= n < j for n in t.note_ticks[-256:])
            t.loops.append((s, j, silent))
        t.ptr = target

    def command(self, t, cmd):
        """The jump table (m4a_tables.c gMPlayJumpTableTemplate + MPlayExtender)."""
        if cmd == 0xB1:                                  # FINE
            self.ply_fine(t)
        elif cmd == 0xB2:                                # GOTO
            self.jump(t, self.word(t.ptr), 'goto')
        elif cmd == 0xB3:                                # PATT
            if len(t.stack) < 3:
                t.stack.append(t.ptr + 4)
                self.jump(t, self.word(t.ptr), 'patt')
            else:
                self.ply_fine(t)
        elif cmd == 0xB4:                                # PEND
            if t.stack:
                t.ptr = t.stack.pop()
        elif cmd == 0xB5:                                # REPT
            n = self.byte(t.ptr)
            if n == 0:
                t.ptr += 1
                self.jump(t, self.word(t.ptr), 'goto')
            else:
                t.rep_n = (t.rep_n + 1) & 0xFF
                t.ptr += 1
                if t.rep_n < n:
                    self.jump(t, self.word(t.ptr), 'rept')
                else:
                    t.rep_n = 0
                    t.ptr += 4
        elif cmd == 0xB9:                                # MEMACC
            self.memacc_cmd(t)
        elif cmd == 0xBA:                                # PRIO
            t.prio = self.read(t)
        elif cmd == 0xBB:                                # TEMPO
            self.tempo_d = self.read(t) * 2
            self.tempo_i = (self.tempo_d * self.tempo_u) >> 8
            self.tempo_log.append((self.clock, self.tempo_d))
        elif cmd == 0xBC:                                # KEYSH
            t.key_shift = s8(self.read(t))
            t.flags |= F_PITCHG
        elif cmd == 0xBD:                                # VOICE
            n = self.read(t)
            t.tone = Tone.read(self.rom, self.voices + 12 * n)
        elif cmd == 0xBE:                                # VOL
            self.vol_args.setdefault(t.idx, set()).add(t.ptr)
            t.vol = self.read(t)
            t.flags |= F_VOLCHG
        elif cmd == 0xBF:                                # PAN
            t.pan = s8(self.read(t) - 0x40)
            t.flags |= F_VOLCHG
        elif cmd == 0xC0:                                # BEND
            t.bend = s8(self.read(t) - 0x40)
            t.flags |= F_PITCHG
        elif cmd == 0xC1:                                # BENDR
            t.bend_range = self.read(t)
            t.flags |= F_PITCHG
        elif cmd == 0xC2:                                # LFOS
            t.lfo_speed = self.read(t)
            if t.lfo_speed == 0:
                self.clear_mod_m(t)
        elif cmd == 0xC3:                                # LFODL
            t.lfo_delay = self.read(t)
        elif cmd == 0xC4:                                # MOD
            t.mod = self.read(t)
            if t.mod == 0:
                self.clear_mod_m(t)
        elif cmd == 0xC5:                                # MODT
            v = self.read(t)
            if v != t.mod_t:
                t.mod_t = v
                t.flags |= F_VOLCHG | F_PITCHG
        elif cmd == 0xC8:                                # TUNE
            t.tune = s8(self.read(t) - 0x40)
            t.flags |= F_PITCHG
        elif cmd == 0xCC:                                # PORT: a sound register write
            self.read(t)
            self.read(t)
            self.note_warn('port', f'track {t.idx}: PORT (a direct sound-register write) ignored')
        elif cmd == 0xCD:                                # XCMD
            self.xcmd(t)
        elif cmd == 0xCE:                                # EOT
            self.ply_endtie(t)
        else:                                            # B6–B8, C6, C7, C9–CB: ply_fine
            self.note_warn(('cmd', cmd), f'track {t.idx}: command {cmd:#x} ends the track (ply_fine)')
            self.ply_fine(t)

    def clear_mod_m(self, t):
        t.mod_m = 0
        t.lfo_speed_c = 0
        t.flags |= F_PITCHG if t.mod_t == 0 else F_VOLCHG

    def memacc_cmd(self, t):
        op = self.read(t)
        a = self.read(t)
        data = self.read(t)
        m = self.memacc
        cur = m[a & 15]
        if op <= 5:
            v = (data, cur + data, cur - data, m[data & 15], cur + m[data & 15], cur - m[data & 15])[op]
            m[a & 15] = v & 0xFF
            return
        ref = data if op <= 11 else m[data & 15]
        k = (op - 6) % 6
        cond = (cur == ref, cur != ref, cur > ref, cur >= ref, cur <= ref, cur < ref)[k] if op <= 17 else None
        if cond is None:
            return
        if cond:
            self.jump(t, self.word(t.ptr), 'memacc')
        else:
            t.ptr += 4

    def xcmd(self, t):
        n = self.read(t)
        tn = t.tone
        if n in (0, 3):                                   # ply_xxx = ply_fine
            self.ply_fine(t)
        elif n == 1:
            tn.wav = self.word(t.ptr)
            t.ptr += 4
        elif n == 2:
            tn.type = self.read(t)
        elif n == 4:
            tn.a = self.read(t)
        elif n == 5:
            tn.d = self.read(t)
        elif n == 6:
            tn.s = self.read(t)
        elif n == 7:
            tn.r = self.read(t)
        elif n == 8:
            t.echo_vol = self.read(t)
        elif n == 9:
            t.echo_len = self.read(t)
        elif n == 10:
            tn.length = self.read(t)
        elif n == 11:
            tn.ps = self.read(t)
        elif n == 12:                                     # xWAIT
            ln = self.byte(t.ptr) | (self.byte(t.ptr + 1) << 8)
            if t.timer < ln:
                t.timer += 1
                t.ptr -= 2
                t.wait = 1
            else:
                t.timer = 0
                t.ptr += 2
        elif n == 13:
            t.start_off = self.word(t.ptr)
            t.ptr += 4
        else:
            self.note_warn(('xcmd', n), f'track {t.idx}: extended command {n} unknown; track ended')
            self.ply_fine(t)

    def ply_endtie(self, t):
        b = self.byte(t.ptr)
        if b < 0x80:
            t.key = b
            t.ptr += 1
        key = t.key
        for ch in t.chans:
            if (ch.status & (SF_START | SF_ENV)) and not (ch.status & SF_STOP) and ch.midi_key == key:
                ch.status |= SF_STOP
                self.end_note(ch, self.clock, 'eot')
                break

    # ── ply_note ───────────────────────────────────────────────────────────────
    def ply_note(self, t, idx):
        t.gate = CLOCK[idx]
        b = self.byte(t.ptr)
        if b < 0x80:
            t.key = b
            t.ptr += 1
            b = self.byte(t.ptr)
            if b < 0x80:
                t.vel = b
                t.ptr += 1
                b = self.byte(t.ptr)
                if b < 0x80:
                    t.gate += b
                    t.ptr += 1
        rpan = 0
        tone = t.tone
        drum = False
        if tone.type & (T_RHY | T_SPL):
            k = t.key
            if tone.type & T_SPL:
                k = self.byte(tone_split_table(tone) + t.key)
            sub = Tone.read(self.rom, tone.wav + 12 * k)
            if sub.type & (T_SPL | T_RHY):
                return
            if tone.type & T_RHY:
                drum = True
                if sub.ps & 0x80:
                    rpan = (sub.ps - 0xC0) * 2
                pkey = sub.key
            else:
                pkey = t.key
        else:
            sub = tone
            pkey = t.key
        prio = min(255, self.prio + t.prio)
        cgb = sub.type & T_CGB
        if cgb:
            ch = self.cgbs[cgb - 1]
            if (ch.status & SF_ON) and not (ch.status & SF_STOP):
                if ch.prio > prio:
                    self.dropped += 1
                    return
                if ch.prio == prio:
                    ti = ch.track.idx if ch.track is not None else -1
                    if not (ti >= t.idx):
                        self.dropped += 1
                        return
        else:
            ch = self.alloc_ds(t, prio)
            if ch is None:
                self.dropped += 1
                return
        # ClearChain + link to this track
        if ch.status:
            self.end_note(ch, self.clock, 'stolen')
            self.silence(ch, float(self.clock), True)
        else:
            self.silence(ch, float(self.clock), False)
        if ch.track is not None and ch in ch.track.chans:
            ch.track.chans.remove(ch)
        t.chans.insert(0, ch)
        ch.track = t
        t.lfo_delay_c = t.lfo_delay
        if t.lfo_delay:
            self.clear_mod_m(t)
        self.trk_vol_pit_set(t)
        ch.gate = t.gate
        ch.midi_key = t.key
        ch.vel = t.vel
        ch.prio = prio
        ch.key = pkey
        ch.rpan = rpan
        ch.type = sub.type
        ch.wav = sub.wav
        ch.a, ch.d, ch.s, ch.r = sub.a, sub.d, sub.s, sub.r
        ch.echo_vol, ch.echo_len = t.echo_vol, t.echo_len
        self.chn_vol_set(ch, t)
        key = max(0, pkey + t.key_m)
        if cgb:
            ch.length = sub.length
            ps = sub.ps
            ch.sweep = ps if (not (ps & 0x80) and (ps & 0x70)) else 8
            if cgb == 1 and ch.sweep != 8:
                self.note_warn('sweep', f'track {t.idx}: square 1 pitch sweep {ch.sweep:#x} not carried')
            ch.freq = cgb_reg(cgb, key, t.pit_m)
            ch.len_frames = None
            if ch.length:
                # the length counter: (64 − n)/256 s (square, noise), (256 − n)/256 s (wave)
                secs = ((256 if cgb == 3 else 64) - (ch.length & (0xFF if cgb == 3 else 0x3F))) / 256.0
                ch.len_frames = secs * FPS
        else:
            ch.count = float(t.start_off)
            if t.start_off:
                self.note_warn('startoff', f'track {t.idx}: a sample start offset (xcmd 0D) not carried')
            ch.wave = self.wave(sub.wav)
            ch.freq = 0
        if t.echo_vol and t.echo_len:
            self.note_warn('echo', f'track {t.idx}: pseudo-echo (xIECV/xIECL) not carried')
        ch.status = SF_START
        t.flags &= 0xF0
        # the log
        n = Note()
        n.track = t.idx
        n.key = t.key
        n.vel = t.vel
        n.tone = tone.ident()
        n.sub = sub.ident()
        n.cgb = cgb
        n.drum = drum
        n.kshift = t.key_shift
        n.rpan = rpan
        n.echo = (t.echo_vol, t.echo_len)
        if cgb:
            goal, hw = cgb_goal(ch.left, ch.right)
            n.goal, n.hwpan = goal, hw
        else:
            n.goal, n.hwpan = None, None
        n.start = float(self.clock)
        n.end = None
        n.why = ''
        n.silent = None
        n.cut = False
        ch.last = n
        self.serial += 1
        n.serial = self.serial
        ch.note = n
        self.notes.append(n)
        if not t.note_ticks or t.note_ticks[-1] != self.clock:
            t.note_ticks.append(self.clock)

    def alloc_ds(self, t, prio):
        best_p, best_t, found_stop, chosen = prio, t.idx, False, None
        for ch in self.ds:
            if not (ch.status & SF_ON):
                return ch
            ct = ch.track.idx if ch.track is not None else -1
            if ch.status & SF_STOP:
                if not found_stop:
                    found_stop = True
                    best_p, best_t, chosen = ch.prio, ct, ch
                    continue
            elif found_stop:
                continue
            if ch.prio < best_p:
                best_p, best_t, chosen = ch.prio, ct, ch
            elif ch.prio > best_p:
                continue
            elif ct > best_t:
                best_t, chosen = ct, ch
            elif ct == best_t:
                chosen = ch
        return chosen

    # ── MPlayMain ──────────────────────────────────────────────────────────────
    def mplay_main(self):
        if self.status < 0:
            return
        self.tempo_c += self.tempo_i
        while self.tempo_c >= 150:
            active = 0
            for t in self.tracks:
                if not (t.flags & F_EXIST):
                    continue
                active = 1
                for ch in list(t.chans):
                    if ch.status & SF_ON:
                        if ch.gate:
                            ch.gate -= 1
                            if ch.gate == 0:
                                ch.status |= SF_STOP
                                self.end_note(ch, self.clock, 'gate')
                    else:
                        t.chans.remove(ch)
                        ch.track = None
                if t.flags & F_START:
                    t.clear()
                    t.flags = F_EXIST
                    t.tone = Tone()
                    t.tone.type, t.tone.key, t.tone.length, t.tone.ps, t.tone.wav = 1, 0, 0, 0, 0
                    t.tone.a = t.tone.d = t.tone.s = t.tone.r = 0
                    t.tone.addr = 0
                    t.chans = []
                guard = 0
                gone = False
                while t.wait == 0:
                    guard += 1
                    if guard > 10000:
                        self.warn(f'track {t.idx}: no wait in 10000 commands at {t.ptr:#x}; stopped')
                        self.ply_fine(t)
                        gone = True
                        break
                    if t.ptr not in t.visited:
                        t.visited[t.ptr] = self.clock
                    b = self.byte(t.ptr)
                    if b < 0x80:
                        cmd = t.running
                    else:
                        t.ptr += 1
                        cmd = b
                        if b >= 0xBD:
                            t.running = b
                    if cmd >= 0xCF:
                        self.ply_note(t, cmd - 0xCF)
                    elif cmd > 0xB0:
                        self.command(t, cmd)
                        if t.flags == 0:
                            gone = True
                            break
                    elif cmd >= 0x80:
                        t.wait = CLOCK[cmd - 0x80]
                    else:                       # running status that is not a note or command
                        self.warn(f'track {t.idx}: data byte {b:#x} with no running status at {t.ptr:#x}; stopped')
                        self.ply_fine(t)
                        gone = True
                        break
                if gone:
                    continue
                t.wait -= 1
                if t.lfo_speed and t.mod:
                    if t.lfo_delay_c:
                        t.lfo_delay_c -= 1
                    else:
                        t.lfo_speed_c = (t.lfo_speed_c + t.lfo_speed) & 0xFF
                        v = t.lfo_speed_c
                        if ((v - 0x40) & 0xFF) < 0x80:
                            r2 = 128 - v
                        else:
                            r2 = s8(v)
                        m = s8((t.mod * r2) >> 6)
                        if m != t.mod_m:
                            t.mod_m = m
                            t.flags |= F_PITCHG if t.mod_t == 0 else F_VOLCHG
            self.clock += 1
            if not active:
                self.status = -1
                return
            self.tempo_c -= 150
        for t in self.tracks:
            if (t.flags & F_EXIST) and (t.flags & (F_VOLCHG | F_PITCHG)):
                self.trk_vol_pit_set(t)
                for ch in list(t.chans):
                    if not (ch.status & SF_ON):
                        t.chans.remove(ch)
                        ch.track = None
                        continue
                    if t.flags & F_VOLCHG:
                        self.chn_vol_set(ch, t)
                    if t.flags & F_PITCHG and ch.cgb:
                        ch.freq = cgb_reg(ch.cgb, max(0, ch.key + t.key_m), t.pit_m)
                t.flags &= 0xF0

    # ── CgbSound (envelopes only: for when the channel frees) ─────────────────
    def cgb_osc_off(self, ch):
        pass

    def cgb_mod_vol(self, ch):
        ch.goal, _hw = cgb_goal(ch.left, ch.right)
        ch.sus_goal = (ch.goal * ch.s + 15) >> 4

    def cgb_sound(self):
        if self.c15:
            self.c15 -= 1
        else:
            self.c15 = 14
        for ch in self.cgbs:
            if not (ch.status & SF_ON):
                continue
            prev_c15 = self.c15
            if ch.len_frames is not None and ch.status:
                ch.len_frames -= 1
                if ch.len_frames <= 0:          # the hardware length counter silenced it
                    self.end_note(ch, self.pos_frame(), 'length')
            if ch.status & SF_START:
                if ch.status & SF_STOP:
                    self.cgb_off(ch)
                    continue
                ch.status = 3
                self.cgb_mod_vol(ch)
                ch.counter = ch.a
                if ch.a & 0xFF:
                    ch.env = 0
                    self.cgb_step_complete(ch, prev_c15)
                else:
                    if self.cgb_decay_start(ch):
                        continue
                    self.cgb_step_complete(ch, prev_c15)
                continue
            if ch.status & SF_IEC:
                ch.echo_len = (ch.echo_len - 1) & 0xFF
                if s8(ch.echo_len) <= 0:
                    self.cgb_off(ch)
                continue
            if (ch.status & SF_STOP) and (ch.status & SF_ENV):
                ch.status &= ~SF_ENV & 0xFF
                ch.counter = ch.r
                if ch.r & 0xFF:
                    self.cgb_step_complete(ch, prev_c15)
                else:
                    self.cgb_echo_start(ch)
                continue
            if self.cgb_step_repeat(ch):
                continue
            self.cgb_step_complete(ch, prev_c15)

    def cgb_off(self, ch):
        ch.status = 0
        self.end_note(ch, self.pos_frame(), 'cgb-off')
        self.silence(ch, self.pos_frame(), False)

    def cgb_echo_start(self, ch):
        """→ True when the channel went off."""
        ch.env = ((ch.goal * ch.echo_vol) + 0xFF) >> 8
        if ch.env:
            ch.status |= SF_IEC
            return False
        self.cgb_off(ch)
        return True

    def cgb_decay_start(self, ch):
        ch.status -= 1
        ch.counter = ch.d
        if ch.counter & 0xFF:
            ch.env = ch.goal
            return False
        return self.cgb_sustain_start(ch)

    def cgb_sustain_start(self, ch):
        if ch.s == 0:
            ch.status &= ~SF_ENV & 0xFF
            self.cgb_echo_start(ch)
            return True                           # envelope_complete: no step this frame
        ch.status -= 1
        ch.env = ch.sus_goal
        ch.counter = 7
        return False

    def cgb_step_repeat(self, ch):
        """envelope_step_repeat → True when the channel went off (or stepping ended)."""
        if ch.counter != 0:
            return False
        self.cgb_mod_vol(ch)
        st = ch.status & SF_ENV
        if st == 0:                               # release
            ch.env = (ch.env - 1) & 0xFF
            if s8(ch.env) <= 0:
                self.cgb_echo_start(ch)
                return True
            ch.counter = ch.r
        elif st == 1:                             # sustain
            ch.env = ch.sus_goal
            ch.counter = 7
        elif st == 2:                             # decay
            ch.env = (ch.env - 1) & 0xFF
            if s8(ch.env) <= s8(ch.sus_goal):
                return self.cgb_sustain_start(ch)
            ch.counter = ch.d
        else:                                     # attack
            ch.env = (ch.env + 1) & 0xFF
            if ch.env >= ch.goal:
                return self.cgb_decay_start(ch)
            ch.counter = ch.a
        return False

    def cgb_step_complete(self, ch, prev_c15):
        ch.counter = (ch.counter - 1) & 0xFF
        if prev_c15 == 0:
            if self.cgb_step_repeat(ch):
                return
            ch.counter = (ch.counter - 1) & 0xFF

    # ── the mixer's DirectSound envelope (SoundMainRAM) ───────────────────────
    def ds_mix(self):
        for ch in self.ds:
            st = ch.status
            if not (st & SF_ON):
                continue
            w = ch.wave
            if st & SF_START:
                if st & SF_STOP:
                    ch.status = 0
                    self.end_note(ch, self.pos_frame(), 'ds-off')
                    self.silence(ch, self.pos_frame(), False)
                    continue
                st = 3
                ch.env = 0
                ch.count = max(0.0, w.size - ch.count)
                if w.loop:
                    st |= SF_LOOP
                env = ch.env + ch.a
                if env >= 0xFF:
                    env = 0xFF
                    st -= 1
                ch.env, ch.status = env, st
            else:
                env = ch.env
                if st & SF_IEC:
                    before = ch.echo_len
                    ch.echo_len = (ch.echo_len - 1) & 0xFF
                    if before <= 1:                 # subs 1; bhi (unsigned, borrow = done)
                        ch.status = 0
                        self.end_note(ch, self.pos_frame(), 'echo-end')
                        self.silence(ch, self.pos_frame(), False)
                        continue
                elif st & SF_STOP:
                    env = (env * ch.r) >> 8
                    if env <= ch.echo_vol:
                        env = ch.echo_vol
                        if env == 0:
                            ch.status = 0
                            self.end_note(ch, self.pos_frame(), 'released')
                            self.silence(ch, self.pos_frame(), False)
                            continue
                        st |= SF_IEC
                else:
                    e = st & SF_ENV
                    if e == 2:
                        env = (env * ch.d) >> 8
                        if env <= ch.s:
                            env = ch.s
                            if env == 0:
                                env = ch.echo_vol
                                if env == 0:
                                    ch.status = 0
                                    self.end_note(ch, self.pos_frame(), 'decayed')
                                    self.silence(ch, self.pos_frame(), False)
                                    continue
                                st |= SF_IEC
                            else:
                                st -= 1
                    elif e == 3:
                        env += ch.a
                        if env >= 0xFF:
                            env = 0xFF
                            st -= 1
                ch.env, ch.status = env, st
            # the sample: one-shots end
            t = ch.track
            if ch.type & T_FIX:
                step = self.pcm_per_vblank
            else:
                km = t.key_m if t is not None else 0
                pm = t.pit_m if t is not None else 0
                key = max(0, min(178, ch.key + km))
                step = self.pcm_per_vblank * w.freq * 2 ** ((key - 60 + pm / 256.0) / 12.0) / self.pcm_freq
            ch.count -= step
            if ch.count <= 0:
                if ch.status & SF_LOOP:
                    ln = max(1, w.size - w.loop_start)
                    while ch.count <= 0:
                        ch.count += ln
                else:
                    ch.status = 0
                    self.end_note(ch, self.pos_frame(), 'sample-end')
                    self.silence(ch, self.pos_frame(), False)

    # ── a frame ────────────────────────────────────────────────────────────────
    def frame_step(self):
        self.mplay_main()
        self.cgb_sound()
        self.ds_mix()
        self.log_curves()
        self.frame += 1

    def log_curves(self):
        pos = self.pos_frame()
        for t in self.tracks:
            if not (t.flags & F_EXIST):
                continue
            v = (t.x, t.vol_mr, t.vol_ml, t.key_shift)
            if self._last_curve.get(t.idx) != v:
                self._last_curve[t.idx] = v
                self.curves[t.idx].append((pos,) + v)
            # the PSG note this track sounds now: its goal follows the volume
            # (CgbModVol at each envelope step; taken at once here)
            for ch in t.chans:
                if ch.cgb and (ch.status & SF_ON):
                    g = cgb_goal(ch.left, ch.right) + (ch.cgb,)
                    if self._last_cgb.get(t.idx) != g:
                        self._last_cgb[t.idx] = g
                        self.cgb_curves.setdefault(t.idx, []).append((pos,) + g)
                    break

    # ── the whole song ─────────────────────────────────────────────────────────
    def music_tracks(self):
        return [t for t in self.tracks if (t.flags & F_EXIST) and not t.idle()]

    def loop_window(self):
        live = self.music_tracks()
        if not live:
            return None
        starts, lens = [], []
        for t in live:
            loops = [l for l in t.loops if not l[2]]
            if not loops:
                return None
            s, j, _ = loops[0]
            starts.append(s)
            lens.append(j - s)
        if any(l <= 0 for l in lens):
            return None
        L = 1
        for l in lens:
            L = L * l // math.gcd(L, l)
        S = max(starts)
        done = [t.note_ticks[-1] for t in self.tracks if t not in live and t.note_ticks]
        if done and max(done) >= S:
            S += ((max(done) - S) // L + 1) * L
        return S, S + L, lens


def tone_split_table(tone):
    return tone.a | (tone.d << 8) | (tone.s << 16) | (tone.r << 24)


def cgb_goal(left, right):
    """CgbModVol: → (envelope goal, hardware pan 'L' | 'R' | 'C')."""
    if right >= left:
        if right // 2 >= left:
            return min(15, (left + right) // 16), 'R'
    elif left // 2 >= right:
        return min(15, (left + right) // 16), 'L'
    return (left + right) // 16, 'C'


def simulate(rom, m4a, header, max_tracks, max_seconds=900.0, warn=print):
    """Run the song. → (player, loop): loop = (start_tick, end_tick) or None;
    player.end_tick is where the song (or its first pass of the loop) ends."""
    p = Player(rom, m4a, header, max_tracks, warn)
    max_frames = int(max_seconds * FPS)
    loop = None
    while p.frame < max_frames:
        p.frame_step()
        if loop is None:
            w = p.loop_window()
            if w is not None:
                S, E, lens = w
                if E - S > 16 * max(lens):
                    warn(f'loop lengths {lens} have an unwieldy common multiple; looping on the longest')
                    E = S + max(lens)
                loop = (S, E)
        if loop is not None and p.clock > loop[1]:
            break
        if loop is None and p.status < 0 and not any(c.status for c in p.ds + p.cgbs):
            break
    else:
        warn(f'stopped after {max_seconds:.0f} s without a loop or an end')
    end = loop[1] if loop else max(1, int(math.ceil(p.pos_frame())) + 1)
    for c in p.ds + p.cgbs:
        if c.note is not None:
            p.end_note(c, float(end), 'song-end')
        p.silence(c, float(end), False)
    p.end_tick = end
    p.order = [t.idx for t in p.tracks]
    return p, loop
