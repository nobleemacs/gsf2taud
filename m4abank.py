"""m4abank.py — the voices a song played, as a SoundFont 2, for midi2taud.

One preset per VARIANT. A DirectSound variant is a ToneData as the track had it
(the voicegroup entry with any xATTA/xDECA/xSUST/xRELE/xWAVE/xTYPE overrides);
a CGB (PSG) variant is that and the envelope GOAL and hardware pan its notes
started with, because a CGB envelope runs in level steps (m4a.c CgbSound): its
times, its sustain and its level all follow the note's volume.

Zones. A plain DirectSound tone is one zone over every key (root 60: the driver
plays a WaveData at freq/1024 Hz on key 60, MidiKeyToFreq). Anything else gets
a zone per key the song used: a drum table (TONEDATA_TYPE_RHY) or a key split
(_SPL) resolves its sub-tone per key, as ply_note does; a drum plays at ITS
key (ToneData.key) with its pan (pan_sweep ≥ 0xC0); a fixed-frequency sample
(TONEDATA_TYPE_FIX) plays at the mixer's rate on every key (scaleTuning 0); a
CGB tone plays at the pitch its register gives that key (MidiKeyToCgbFreq: the
11-bit period rounds, which is audible at the top).

Samples:
  * DirectSound: the WaveData's signed 8-bit PCM ×256, at freq/1024 Hz, looped
    when status bit 14 says so (WAVE_DATA_FLAG_LOOP); DPCM-compressed data
    (wav.type ≠ 0 on a CMP/REV tone) decoded with gDeltaEncodingTable; a REV
    tone's data reversed.
  * Square (CGB 1, 2): the four duties, 8 steps a cycle, DC removed.
  * Wave (CGB 3): the tone's 32 4-bit samples, DC removed. It sounds an octave
    below a square on the same register (65536 vs 131072 Hz ÷ (2048 − x)).
  * Noise (CGB 4): the GB LFSR, 15-bit (32767 steps) or 7-bit (127), one step a
    sample at 32768 Hz; each key's zone is tuned to its clock from gNoiseTable.
  Levels: a CGB channel at level 15 swings 240 of mGBA's units (level × 16,
  unipolar), DirectSound full scale ±508 (sample × 4). SF2 ±1.0 is 508 units for
  DirectSound and 240 for CGB (the CGB samples swing ±(u − mean)); the
  remaining −6.5 dB is in the notes' velocity (gsf2taud.py).

Envelopes:
  * DirectSound (SoundMainRAM, a step a frame, 59.73 Hz): the attack ADDS
    `a` to a 0…255 level (linear, as SF2's); decay and release MULTIPLY by
    d/256, r/256 (linear in dB, as SF2's); sustain is s/255. A decay that
    reaches its sustain within DECAY_FLOOR is given that long (midi2taud draws a
    shorter decay as zero-length nodes, which a Taud envelope reads as its end).
  * CGB (CgbSound, 63.7 steps a second): every step count `a` / `d` / `r` the
    level moves ONE of its 16 steps — linear in amplitude. Attack over
    goal·a steps (exact in SF2). Decay from the goal to sustainGoal =
    (goal·s + 15) >> 4 over (goal − sustainGoal)·d steps: SF2's decay is
    timed to reach the same level at the same time; to silence, it crosses
    −20 dB where the linear fall does. Release over sustainGoal·r steps: Taud
    fades linearly in amplitude over a quarter of SF2's release time
    (midi2taud _zone_fadeout), so the SF2 release is four times that.
"""

import math
import struct

from m4aplay import (FPS, T_CGB, T_FIX, T_REV, T_CMP, T_SPL, T_RHY, Tone, cgb_reg, cgb_hz)
from gbarom import ROM_BASE, u32

TC_MIN, TC_MAX = -12000, 8000
DECAY_FLOOR = 0.006
CGB_STEP = FPS * 16.0 / 15.0
DELTA = [0, 1, 4, 9, 16, 25, 36, 49, -64, -49, -36, -25, -16, -9, -4, -1]
DUTY = [[0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 0]]
SQ_SPS = 8
SQ_BASE = 440.0
WAVE_SPS = 4
WAVE_BASE = 220.0
NOISE_RATE = 32768
WAVE_VOL = {0x00: 0.0, 0x60: 0.25, 0x40: 0.5, 0x80: 0.75, 0x20: 1.0}


def _tc(seconds):
    if seconds <= 0.0011:
        return TC_MIN
    return max(TC_MIN, min(TC_MAX, round(1200.0 * math.log2(seconds))))


def _sus_cb(level):
    if level <= 0:
        return 1440
    return max(0, min(1440, round(-200.0 * math.log10(level))))


# ── DirectSound envelope ─────────────────────────────────────────────────────
def _int_fall(start, mul, floor_level=0, limit=100000):
    """The mixer's decay/release, a frame at a time: env = env·mul >> 8 (the
    floor speeds the fall near the bottom — at env 40 a d of 249 falls 0.45 dB a
    frame, not 0.24). → the levels after each frame until env ≤ floor_level."""
    out = []
    env = start
    while env > floor_level and len(out) < limit:
        env = (env * mul) >> 8
        out.append(env)
    return out


def ds_env(a, d, s, r):
    """→ (attack tc, decay tc, sustain cB, release tc), the times from the
    driver's own integer steps (SoundMainRAM):
      * decay to a sustain: SF2's decay reaches it on the frame the driver
        does; to silence (s = 0): the same ENERGY as the driver's fall —
        midi2taud draws SF2's 100 dB fall, whose energy is T/(2·ln 10^5);
      * release from the sustain level (half scale when there is none): the
        same energy as the driver's fall — Taud fades it linearly over a quarter
        of SF2's time (midi2taud _zone_fadeout), energy level²·T/3."""
    if a == 0:
        att = TC_MAX                       # never rises: the driver adds 0 a frame
    else:
        att = _tc(max(0.0, 255.0 / a - 1.0) / FPS)
    sus = _sus_cb(s / 255.0)
    if s >= 255:
        dec = TC_MIN
    elif s > 0:
        n = len(_int_fall(255, d, s)) if d else 0
        t = max(n / FPS, DECAY_FLOOR)
        dec = _tc(t * 1000.0 / min(sus, 1000))
    else:
        steps = _int_fall(255, d) if d else []
        e = (1.0 + sum((v / 255.0) ** 2 for v in steps)) / FPS
        dec = _tc(max(e * 2.0 * math.log(1e5), DECAY_FLOOR * 4.5))
    if r == 0:
        rel = _tc(1.0 / FPS)
    else:
        e0 = s if s > 0 else 128
        steps = _int_fall(e0, r)
        e = (1.0 + sum((v / e0) ** 2 for v in steps)) / FPS
        rel = _tc(4.0 * 3.0 * e)
    return att, dec, sus, rel


# ── CGB envelope ─────────────────────────────────────────────────────────────
def cgb_level(cgb, goal):
    """The hardware level a goal sounds at, 0…1 (wave: its volume code)."""
    if cgb == 3:
        return WAVE_VOL.get(CGB3_VOL_EXT(goal), 0.0)
    return (goal & 15) / 15.0


def CGB3_VOL_EXT(goal):
    from m4aplay import CGB3_VOL
    return CGB3_VOL[goal] if 0 <= goal < 16 else 0x00


def cgb_env(cgb, goal, a, d, s, r):
    """→ (attack tc, decay tc, sustain cB, release tc) for a note started at `goal`."""
    g = goal
    sg = (g * s + 15) >> 4
    lvl = cgb_level(cgb, g)
    slvl = cgb_level(cgb, sg)
    att = TC_MIN if a == 0 else _tc(g * a / CGB_STEP)
    rel_from = sg if sg else g / 2.0
    if d == 0:
        if sg >= g:
            dec, sus = TC_MIN, 0
        elif sg == 0:
            dec, sus = _tc(DECAY_FLOOR * 4.5), 1440
        else:
            sus = _sus_cb(slvl / lvl) if lvl else 0
            dec = _tc(DECAY_FLOOR * 1000.0 / max(1, min(sus, 1000)))
    else:
        td = max(0, g - sg) * d / CGB_STEP
        if sg >= g or td <= 0:
            dec, sus = TC_MIN, 0
        elif sg == 0:
            dec, sus = _tc(max(td, DECAY_FLOOR) * 4.5), 1440
        else:
            sus = _sus_cb(slvl / lvl) if lvl else 0
            if sus <= 0:
                dec = TC_MIN
            else:
                dec = _tc(max(td, DECAY_FLOOR) * 1000.0 / min(sus, 1000))
    if r == 0:
        rel = _tc(1.0 / FPS)
    else:
        rel = _tc(4.0 * rel_from * r / CGB_STEP)
    return att, dec, sus, rel


# ── samples ──────────────────────────────────────────────────────────────────
class _Sample:
    __slots__ = ('name', 'data', 'rate', 'corr', 'loop', 'base_hz')


def _rate_of(exact):
    r = max(1, round(exact))
    corr = round(1200.0 * math.log2(exact / r))
    return r, max(-99, min(99, corr))


def _ds_sample(rom, addr, ttype, pcm_freq):
    o = addr - ROM_BASE
    s = _Sample()
    s.name = f'W{addr & 0xFFFFFF:06X}'
    wtype = struct.unpack_from('<H', rom, o)[0]
    loop = bool(rom[o + 3] & 0xC0)
    freq = u32(rom, o + 4) / 1024.0
    ls = u32(rom, o + 8)
    size = u32(rom, o + 12)
    size = min(size, 4 << 20)
    data = []
    if (ttype & (T_CMP | T_REV)) and wtype != 0:
        p = o + 16
        while len(data) < size and p + 33 <= len(rom):
            v = rom[p]
            blk = [v - 256 if v & 0x80 else v]
            b = rom[p + 1]
            v = (blk[0] + DELTA[b & 15])
            blk.append(v)
            for i in range(31):
                b = rom[p + 2 + i]
                v = v + DELTA[b >> 4]
                blk.append(v)
                v = v + DELTA[b & 15]
                blk.append(v)
            data += [((x + 128) & 0xFF) - 128 for x in blk]
            p += 33
        data = data[:size]
    else:
        raw = rom[o + 16:o + 16 + size]
        data = [x - 256 if x & 0x80 else x for x in raw]
    if ttype & T_REV:
        data = data[::-1]
        loop = False
    s.data = [x * 256 for x in data] or [0, 0]
    s.rate, s.corr = _rate_of(pcm_freq if ttype & T_FIX else freq)
    s.loop = (ls, len(s.data)) if loop and len(s.data) - ls >= 2 and ls < len(s.data) else None
    return s


def _square_sample(duty):
    s = _Sample()
    s.name = f'Square duty {duty}'
    pat = DUTY[duty & 3]
    mean = sum(pat) / 8.0
    cyc = []
    for u in pat:
        cyc += [round((u - mean) * 32767)] * SQ_SPS
    s.data = cyc * 4
    s.rate, s.corr = round(SQ_BASE * 8 * SQ_SPS), 0
    s.base_hz = s.rate / (8.0 * SQ_SPS)
    s.loop = (len(cyc), len(s.data))
    return s


def _wave_sample(rom, addr):
    s = _Sample()
    o = addr - ROM_BASE
    raw = rom[o:o + 16] if 0 <= o <= len(rom) - 16 else bytes(16)
    nib = []
    for b in raw:
        nib += [b >> 4, b & 15]
    mean = sum(nib) / 32.0
    s.name = f'Wave {addr & 0xFFFFFF:06X}'
    cyc = []
    for n in nib:
        cyc += [round((n - mean) / 15.0 * 32767)] * WAVE_SPS
    s.data = cyc * 4
    s.rate, s.corr = round(WAVE_BASE * 32 * WAVE_SPS), 0
    s.base_hz = s.rate / (32.0 * WAVE_SPS)
    s.loop = (len(cyc), len(s.data))
    return s


def _noise_sample(short):
    s = _Sample()
    s.name = 'Noise 7-bit' if short else 'Noise 15-bit'
    x = 0x7FFF
    out = []
    for _ in range(127 if short else 32767):
        u = 1 if not (x & 1) else 0
        bit = (x ^ (x >> 1)) & 1
        x = (x >> 1) | (bit << 14)
        if short:
            x = (x & ~0x40) | (bit << 6)
        out.append(round((u - 0.5) * 32767))
    s.data = out
    s.rate, s.corr = NOISE_RATE, 0
    s.base_hz = float(NOISE_RATE)
    s.loop = (0, len(out))
    return s


def _name20(s):
    return s.encode('ascii', 'replace')[:20].ljust(20, b'\0')


def _tune(cents):
    cents = round(cents)
    coarse = max(-120, min(120, round(cents / 100.0)))
    fine = max(-99, min(99, cents - coarse * 100))
    return coarse, fine


def build_sf2(path, variants, rom, pcm_freq, title='GBA bank', comment=''):
    """variants: [(variant key, (bank, prog), keys, release cap s | None)], the
    key (tone tuple,) for DirectSound or (tone tuple, goal, hwpan) for CGB. Writes `path`; → the
    number of presets written."""
    samples = []
    sindex = {}

    def sample(key, make):
        if key not in sindex:
            sindex[key] = len(samples)
            samples.append(make())
        return sindex[key]

    insts = []
    presets = []
    for vkey, (bank, prog), keys, cap in variants:
        # a release the driver usually cuts short: Taud fades a release out
        # linearly over a quarter of SF2's time (midi2taud _zone_fadeout), so
        # the fade ends where the cut fell
        cap_tc = _tc(4.0 * cap) if cap is not None else None
        tt = vkey[0]
        tone = Tone()
        tone.type, tone.key, tone.length, tone.ps, tone.wav, tone.a, tone.d, tone.s, tone.r = tt
        goal = vkey[1] if len(vkey) > 1 else None
        zones = []
        plain_ds = not (tone.type & (T_RHY | T_SPL)) and not (tone.type & T_CGB) and not (tone.type & T_FIX)
        for k in (sorted(keys) if not plain_ds else [None]):
            if tone.type & (T_RHY | T_SPL):
                idx = k
                if tone.type & T_SPL:
                    tbl = tone.a | (tone.d << 8) | (tone.s << 16) | (tone.r << 24)
                    idx = rom[tbl - ROM_BASE + k]
                sub = Tone.read(rom, tone.wav + 12 * idx)
                pk = sub.key if tone.type & T_RHY else k
            else:
                sub = tone
                pk = k
            cgb = sub.type & T_CGB
            lo, hi = (0, 127) if k is None else (k, k)
            gens = [(43, lo | (hi << 8))]
            if cgb:
                if cgb in (1, 2):
                    si = sample(('sq', sub.wav & 3), lambda: _square_sample(sub.wav & 3))
                elif cgb == 3:
                    si = sample(('wave', sub.wav), lambda: _wave_sample(rom, sub.wav))
                else:
                    short = bool(sub.wav & 1)
                    si = sample(('noise', short), lambda: _noise_sample(short))
                hz = cgb_hz(cgb, cgb_reg(cgb, pk))
                if cgb == 4:
                    hz = hz                        # the LFSR clock: one step a sample
                cents = 1200.0 * math.log2(hz / samples[si].base_hz)
                coarse, fine = _tune(cents)
                att, dec, sus, rel = cgb_env(cgb, goal or 0, sub.a & 7, sub.d & 7, sub.s & 15, sub.r & 7)
                if cap_tc is not None:
                    rel = min(rel, cap_tc)
                gens += [(34, att), (36, dec), (37, sus), (38, rel), (54, 1),
                         (51, coarse), (52, fine), (58, k)]
            else:
                if not sub.wav or not (ROM_BASE <= sub.wav < ROM_BASE + len(rom)):
                    continue
                si = sample(('ds', sub.wav, sub.type & (T_FIX | T_CMP | T_REV)),
                            lambda: _ds_sample(rom, sub.wav, sub.type, pcm_freq))
                smp = samples[si]
                att, dec, sus, rel = ds_env(sub.a, sub.d, sub.s, sub.r)
                if cap_tc is not None:
                    rel = min(rel, cap_tc)
                gens += [(34, att), (36, dec), (37, sus), (38, rel), (54, 1 if smp.loop else 0)]
                if k is None:
                    gens += [(58, 60)]
                elif sub.type & T_FIX:
                    gens += [(56, 0), (58, k)]
                else:
                    coarse, fine = _tune((pk - 60) * 100.0)
                    gens += [(51, coarse), (52, fine), (58, k)]
                if (tone.type & T_RHY) and (sub.ps & 0x80):
                    rp = (sub.ps - 0xC0) * 2
                    th = math.atan2(128 + rp, 127 - rp)          # the driver's linear law
                    pan8 = th * 512.0 / math.pi
                    gens.insert(1, (17, max(-500, min(500, round((pan8 - 127.5) * 1000.0 / 255.0)))))
            gens.append((53, si))
            zones.append(gens)
        if not zones:
            continue
        kind = 'drums' if tone.type & T_RHY else 'split' if tone.type & T_SPL else \
            ('cgb%d' % (tone.type & 7)) if tone.type & T_CGB else 'pcm'
        nm = f'{kind} {(tone.wav & 0xFFFFFF):06X}'
        if goal is not None:
            nm += f' g{goal}{vkey[2]}'
        insts.append((nm, zones))
        presets.append((nm, bank, prog, len(insts) - 1))

    smpl = bytearray()
    shdr = bytearray()
    for s in samples:
        start = len(smpl) // 2
        smpl += struct.pack(f'<{len(s.data)}h', *[max(-32768, min(32767, v)) for v in s.data])
        smpl += bytes(46 * 2)
        end = start + len(s.data)
        ls, le = (start + s.loop[0], start + s.loop[1]) if s.loop else (start, end)
        shdr += _name20(s.name) + struct.pack('<IIIIIBbHH', start, end, ls, le, s.rate,
                                              60, s.corr, 0, 1)
    shdr += _name20('EOS') + bytes(26)

    phdr = bytearray(); pbag = bytearray(); pgen = bytearray()
    for i, (nm, bank, prog, ii) in enumerate(presets):
        phdr += _name20(nm) + struct.pack('<HHHIII', prog, bank, i, 0, 0, 0)
        pbag += struct.pack('<HH', len(pgen) // 4, 0)
        pgen += struct.pack('<HH', 41, ii)
    phdr += _name20('EOP') + struct.pack('<HHHIII', 0, 0, len(presets), 0, 0, 0)
    pbag += struct.pack('<HH', len(pgen) // 4, 0)
    pgen += bytes(4)
    inst = bytearray(); ibag = bytearray(); igen = bytearray()
    nbag = 0
    for nm, zones in insts:
        inst += _name20(nm) + struct.pack('<H', nbag)
        for gens in zones:
            ibag += struct.pack('<HH', len(igen) // 4, 0)
            nbag += 1
            for op, amt in gens:
                if op in (43, 53, 54, 58):
                    igen += struct.pack('<HH', op, amt)
                else:
                    igen += struct.pack('<Hh', op, amt)
    inst += _name20('EOI') + struct.pack('<H', nbag)
    ibag += struct.pack('<HH', len(igen) // 4, 0)
    igen += bytes(4)

    def chunk(cid, data):
        data = bytes(data)
        return cid + struct.pack('<I', len(data)) + data + (b'\0' if len(data) & 1 else b'')

    def lst(kind, parts):
        body = kind + b''.join(parts)
        return b'LIST' + struct.pack('<I', len(body)) + body

    def zstr(s):
        b = s.encode('latin-1', 'replace') + b'\0'
        return b + (b'\0' if len(b) & 1 else b'')

    info = lst(b'INFO', [chunk(b'ifil', struct.pack('<HH', 2, 1)),
                         chunk(b'isng', zstr('EMU8000')),
                         chunk(b'INAM', zstr(title[:250])),
                         chunk(b'ICMT', zstr(('gsf2taud: ' + comment)[:250]))])
    sdta = lst(b'sdta', [chunk(b'smpl', smpl)])
    pdta = lst(b'pdta', [chunk(b'phdr', phdr), chunk(b'pbag', pbag), chunk(b'pmod', bytes(10)),
                         chunk(b'pgen', pgen), chunk(b'inst', inst), chunk(b'ibag', ibag),
                         chunk(b'imod', bytes(10)), chunk(b'igen', igen), chunk(b'shdr', shdr)])
    body = b'sfbk' + info + sdta + pdta
    with open(path, 'wb') as f:
        f.write(b'RIFF' + struct.pack('<I', len(body)) + body)
    return len(presets)
