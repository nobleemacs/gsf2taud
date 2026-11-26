#!/usr/bin/env python3
"""gsf2taud.py — Game Boy Advance music (.gsf / .minigsf) to Taud. EXPERIMENTAL, out of tree.

Usage:
    python3 gsf2taud.py song.minigsf [out.taud] [--song N] [--seconds S]
                        [--no-vibrato] [-v] [-- midi2taud options…]
    python3 gsf2taud.py some_dir/ [out_dir/]          # every .minigsf in it
    python3 gsf2taud.py song.minigsf --list           # the song table

A GSF is the game's ROM: its own code, its own sound driver, and a mini that
patches in the number of the song to start. Every rip here is a game on
Nintendo's MP2K driver ("Sappy", the SDK's m4a library), so nothing is
emulated: the song is run through a port of that driver as pret decompiled it
(m4aplay.py) for its notes, channel allocation and per-frame pitch, volume
and pan; the voices it played become a SoundFont (m4abank.py); and
midi2taud.py — the in-tree converter, unmodified on disk — turns the two into a
.taud. The .sf2 is kept beside the .taud.

What carries over: notes and their lengths (gate times, ties), velocity and
track volume (MP2K's are LINEAR in amplitude, SF2's and Taud's squared: both
are converted), pan (the driver's linear pan law is matched in direction, its
level difference folded into the volume), master volume, pitch (bend, bend
range, tune, key shift, vibrato as bends), volume and pan LFOs, the DirectSound
ADSR and every per-track override, the PSG (CGB) channels — square, wave and
noise, their envelopes at the level each note started at, their hardware pan
and their register pitch — drum tables and key splits, channel allocation and
stealing, loops (as the song's loop) and tempo (on a whole-number Taud BPM; a
tick is 2 or 4 Taud ticks, whichever rounds closer). What does not: reverb
(the mixer's echo), pseudo-echo, square 1's sweep, the PSG envelope's level
STEPS (drawn as ramps), a PSG note's level following a volume change while it
sounds, leading silence (midi2taud starts at the first note), and anything the
game does to the song from outside.
"""

import argparse
import math
import os
import struct
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gbarom import load_gsf, M4A, song_number, ROM_BASE, u32           # noqa: E402
from m4aplay import simulate, FPS, SEQ_PPQN, T_RHY, T_SPL, T_CGB          # noqa: E402
from m4abank import build_sf2, cgb_level                                  # noqa: E402

CONVERTERS = os.path.normpath(HERE)
SUB = 4                      # MIDI ticks per sequence tick
PPQ = SEQ_PPQN * SUB
MIDI_CHANNELS = [c for c in range(64) if c != 9]
CGB_DB = 20.0 * math.log10(240.0 / 508.0)     # a CGB channel's full swing against DirectSound's
CENTRE_DB = 10.0 * math.log10(2.0)            # a CGB channel on both sides
CGB_VEL_DB = CGB_DB + CENTRE_DB
# The mixer's echo, as measured against mGBA (see README): ECHO_TAP frames
# short of the PCM ring's P frames, at ECHO_GAIN × reverb/256. Set from the
# environment while fitting.
ECHO_TAP = float(os.environ.get('GSF_ECHO_TAP', 1.0))
ECHO_GAIN = float(os.environ.get('GSF_ECHO_GAIN', 1.0))


def load_midi2taud():
    """midi2taud.py as a module, its per-channel state array widened from 16 to 64."""
    path = os.path.join(CONVERTERS, 'midi2taud.py')
    src = open(path, encoding='utf-8').read()
    old = 'chs = [_ChState() for _ in range(16)]'
    if src.count(old) != 1:
        sys.exit('error: midi2taud.py changed: cannot widen its channel array')
    src = src.replace(old, 'chs = [_ChState() for _ in range(64)]')
    # A decay leg is drawn as straight lines (in amplitude) between equal-dB
    # points, three of them for any decay under 3 s: a pluck falling 100 dB in
    # 1.5 s gets 33 dB a segment, and a line over 33 dB carries 2.5 times the
    # energy of the exponential — 4 dB too loud (Sapphire's Mount Pyre). At
    # least a segment per 6 dB of depth, within the envelope's 25 nodes.
    old = '            n = _env_seg_count(edec)'
    if src.count(old) != 1:
        sys.exit('error: midi2taud.py changed: cannot deepen its decay segments')
    src = src.replace(old, '            n = max(_env_seg_count(edec), min(18, math.ceil(sus_cb / 60.0)))')
    if CONVERTERS not in sys.path:
        sys.path.insert(0, CONVERTERS)
    mod = types.ModuleType('midi2taud')
    mod.__file__ = path
    sys.modules['midi2taud'] = mod
    exec(compile(src, path, 'exec'), mod.__dict__)
    return mod


HWPAN = {'L': 0, 'R': 127, 'C': 64}
# A PSG note's velocity is its channel's full swing on both sides (−6.5 + 3 dB);
# CC7 carries its level now — the goal's, which follows the track's volume —
# and 3 dB less when the hardware pans it to one side.
CGB_VEL = max(1, min(127, round(127.0 * 10.0 ** (CGB_VEL_DB / 40.0))))


def cgb_cc7(cgb, goal, hw):
    lvl = cgb_level(cgb, goal)
    if lvl <= 0:
        return 0
    return cc_of_db(20.0 * math.log10(lvl) - (0.0 if hw == 'C' else CENTRE_DB))


def ds_velocity(v):
    """MP2K: amplitude ∝ v; SF2/Taud: ∝ (vel/127)²."""
    return max(1, min(127, round(127.0 * math.sqrt(v / 127.0)))) if v > 0 else 0


def cc_of_db(db):
    return max(0, min(127, round(127.0 * 10.0 ** (db / 40.0))))


def pan_cc(right, left):
    """The pan whose constant-power law gives this right/left ratio (Taud §11.3)."""
    if right == 0 and left == 0:
        return 64
    th = math.atan2(right, left)                 # 0 = left … π/2 = right
    return max(0, min(127, round(th * 512.0 / math.pi * 127.0 / 255.0)))


def build_events(p, loop, title, master_volume, vibrato=True, warn=print):
    used = []
    for tid in p.order:
        if any(n.track == tid for n in p.notes):
            used.append(tid)
    chans = []
    for tid in used:
        if any(n.track == tid and not n.cgb for n in p.notes):
            chans.append((tid, 'ds'))
        if any(n.track == tid and n.cgb for n in p.notes):
            chans.append((tid, 'cgb'))
    if len(chans) > len(MIDI_CHANNELS):
        sys.exit('error: more parts than MIDI channels')
    ch_of = {c: MIDI_CHANNELS[i] for i, c in enumerate(chans)}
    names = {ch_of[c]: f'Track {c[0]}' + (' PSG' if c[1] == 'cgb' else '') for c in chans}

    def fold(n):
        return not (n.tone[0] & (T_RHY | T_SPL))

    def midi_key(n):
        return n.key + (n.kshift if fold(n) else 0)

    # which channels count the key shift into their notes (melodic) or their bend (drums)
    base_ks = {}
    for c in chans:
        ns = [n for n in p.notes if n.track == c[0] and (n.cgb != 0) == (c[1] == 'cgb')]
        folded = sum(1 for n in ns if fold(n))
        base_ks[c] = folded * 2 >= len(ns)
        if 0 < folded < len(ns) and any(n.kshift for n in ns):
            warn(f'track {c[0]}: drums and melody share a key shift; the drums follow the melody')

    # variants → (bank, prog)
    vkeys = []
    vkeyset = {}
    keys_of = {}
    for n in p.notes:
        vk = vkey(n)
        if vk not in vkeyset:
            vkeyset[vk] = len(vkeys)
            vkeys.append(vk)
        keys_of.setdefault(vk, set()).add(midi_key(n))
    variants = [(vk, (i // 128, i % 128), keys_of[vk], release_cap(p, [n for n in p.notes if vkey(n) == vk]))
                for i, vk in enumerate(vkeys)]
    slot = {v[0]: v[1] for v in variants}

    ev = []

    def put(pos, order, e):
        ev.append((max(0, round(pos * SUB)), order, e))

    # DirectSound sounds a frame after the PSG: the mixer fills the PCM slot
    # the DMA plays from the NEXT frame (m4a_1.s SoundMain: the slot after the
    # one pcmDmaCounter says is playing), while CgbSound writes the PSG's
    # registers at once. A frame is tempo/150 sequence ticks.
    tempos = sorted((t, v) for t, v in p.tempo_log if v > 0)

    def ds_late(pos):
        v = 150
        for t, tv in tempos:
            if t <= pos:
                v = tv
            else:
                break
        return pos + v / 150.0

    put(0, 0, ('title', title))
    tempo_at = {}
    for pos, tempo in p.tempo_log:
        if tempo > 0:
            tempo_at[pos] = tempo
    for pos, tempo in sorted(tempo_at.items()):
        put(pos, 0, ('tempo', 60.0 * tempo * FPS / (150.0 * SEQ_PPQN)))
    if loop:
        put(loop[0], 0, ('loopstart',))
        put(loop[1], 0, ('loopend',))

    mv_db = 20.0 * math.log10((master_volume + 1) / 16.0)
    for c in chans:
        ch = ch_of[c]
        curve = p.curves.get(c[0], [])
        bends = [(x / 256.0 - (ks if base_ks[c] else 0)) for _pos, x, _r, _l, ks in curve]
        mx = max([abs(b) for b in bends] + [0])
        rng = max(2, min(127, int(math.ceil(mx)) + (0 if mx == int(mx) else 0)))
        if mx > 127:
            warn(f'track {c[0]}: pitch offsets reach {mx:.0f} semitones; clipped')
        for num, val in ((101, 0), (100, 0), (6, rng), (38, 0)):
            put(0, 1, ('cc', ch, num, val))
        if c[1] == 'cgb':
            for pos, goal, hw, cgb in p.cgb_curves.get(c[0], []):
                put(pos, 1, ('cc', ch, 7, cgb_cc7(cgb, goal, hw)))
                put(pos, 1, ('cc', ch, 10, HWPAN[hw]))
        last = [None] * 3
        for (pos, x, r, l, ks), b in zip(curve, bends):
            if not vibrato:
                pass
            b14 = max(0, min(16383, 8192 + round(b / rng * 8192)))
            vals = [('bend', b14)]
            if c[1] == 'ds':
                pw = (r * r + l * l) / 65536.0
                c7 = 0 if pw <= 0 else cc_of_db(10.0 * math.log10(pw) + mv_db)
                vals += [(7, c7), (10, pan_cc(r, l))]
            # the curves stay on their frames: shifted with the notes, a volume
            # change can land in the row of a delayed note, and midi2taud then
            # sounds the note before the change (Journey of Fugitives' opening
            # fade-in played its first note loud)
            at = pos
            for i, (k, v) in enumerate(vals):
                if last[i] != v:
                    last[i] = v
                    put(at, 1, ('bend', ch, v) if k == 'bend' else ('cc', ch, k, v))

    cur_prog = {}
    cur_pan = {}
    active = {}
    ends = {}
    notes = sorted(p.notes, key=lambda n: (n.start, n.serial))
    for n in notes:
        ch = ch_of[(n.track, 'cgb' if n.cgb else 'ds')]
        k = (ch, midi_key(n))
        if k in active and ends[id(active[k])] > n.start:
            ends[id(active[k])] = n.start
        active[k] = n
        ends[id(n)] = n.end if n.end is not None else n.start
    dropped = 0
    silent = 0
    for n in notes:
        mk = midi_key(n)
        if not 1 <= mk <= 127:
            dropped += 1
            continue
        part = (n.track, 'cgb' if n.cgb else 'ds')
        ch = ch_of[part]
        if n.cgb:
            lvl = cgb_level(n.cgb, n.goal)
            if lvl <= 0:
                silent += 1
                continue
            vel = CGB_VEL
            put(n.start, 1, ('cc', ch, 7, cgb_cc7(n.cgb, n.goal, n.hwpan)))
            put(n.start, 1, ('cc', ch, 10, HWPAN[n.hwpan]))
        else:
            vel = ds_velocity(n.vel)
        if vel <= 0:
            silent += 1
            continue
        bank, prog = slot[vkey(n)]
        if cur_prog.get(ch) != (bank, prog):
            cur_prog[ch] = (bank, prog)
            at = n.start if n.cgb else ds_late(n.start)
            put(at, 1, ('cc', ch, 0, bank))
            put(at, 1, ('prog', ch, prog))
        start = n.start
        end = max(ends[id(n)], n.start + 1.0 / SUB)
        if not n.cgb:
            start, end = ds_late(start), ds_late(end)
        put(start, 2, ('on', ch, mk, vel))
        put(end, 0, ('off', ch, mk))
    if dropped:
        warn(f'{dropped} note(s) outside MIDI keys 1…127 dropped')
    put(p.end_tick, 3, ('eot',))
    ev.sort(key=lambda e: (e[0], e[1]))
    merged = [(t, i, e) for i, (t, _o, e) in enumerate(ev)]
    return merged, variants, names


def vkey(n):
    return (n.tone, n.goal, n.hwpan) if n.cgb else (n.tone,)


def tick_seconds(p):
    """→ f(tick) = seconds from the song's start, through its tempo changes."""
    marks = []                                   # (tick, seconds there, seconds a tick)
    sec = 0.0
    last_t, last_rate = 0, 150.0 / (150 * FPS)
    for t, tempo in sorted(p.tempo_log):
        if tempo <= 0:
            continue
        sec += (t - last_t) * last_rate
        last_t, last_rate = t, 150.0 / (tempo * FPS)
        marks.append((t, sec, last_rate))
    if not marks:
        marks = [(0, 0.0, last_rate)]

    def f(tick):
        m = marks[0]
        for mk in marks:
            if mk[0] <= tick:
                m = mk
            else:
                break
        return m[1] + (tick - m[0]) * m[2]
    return f


def release_cap(p, notes):
    """The driver hands a releasing channel to the next note that needs one
    (ply_note prefers them), and the tail is cut on the spot; Taud lets a
    released note fade out on its own. Where at least half of a variant's
    releases are cut, its release is capped at the median time they got —
    → seconds, or None."""
    sec = getattr(p, '_tick_sec', None)
    if sec is None:
        sec = p._tick_sec = tick_seconds(p)
    rel = [n for n in notes if n.end is not None and n.silent is not None]
    if not rel:
        return None
    cut = sorted(sec(n.silent) - sec(n.end) for n in rel if n.cut)
    if len(cut) * 2 < len(rel):
        return None
    return max(0.005, cut[len(cut) // 2])


def pick_grid(p):
    """(rpb, speed). A sequence tick is k Taud ticks, k = 2 or 4, whichever
    rounds the song's tempos to a whole Taud BPM more closely (the BPM is
    k·tempo·59.7275/60, a tick lasting 2.5/BPM s) within the register (≤ 1048).
    Rows are 32nd notes (rpb 8: three sequence ticks): midi2taud's rpb is a
    power of two, and on 24 ticks a quarter no power of two puts a triplet on
    a row, so those ride on note delays whatever the grid."""
    tempos = [t for _, t in p.tempo_log if t > 0] or [150]
    best = None
    for k in (2, 4):
        bpms = [k * t * FPS / 60.0 for t in tempos]
        if max(bpms) > 1048:
            continue
        err = max(abs(round(b) - b) / b for b in bpms)
        if best is None or err < best[0] - 1e-9:
            best = (err, k)
    k = best[1] if best else 1
    return 8, 3 * k


def convert(path, out, args, m2t=None):
    rom, tags, patches = load_gsf(path)
    m4a = M4A(rom)
    song = args.song if args.song is not None else song_number(patches)
    hdr, ms, ntr = m4a.song(song)
    if not (ROM_BASE <= hdr < ROM_BASE + len(rom)) or rom[hdr - ROM_BASE] == 0:
        sys.exit(f'error: song {song} is empty or missing in this ROM')
    title = tags.get('title') or f'Song {song}'
    print(f'{os.path.basename(path)}: song {song} (header {hdr:#x}, player {ms}), '
          f'{rom[hdr - ROM_BASE]} tracks, mixer {m4a.max_chans} channels, master volume {m4a.master_volume}')
    warn = (lambda m: print('  warning: ' + m))
    p, loop = simulate(rom, m4a, hdr, ntr, max_seconds=args.seconds, warn=warn)
    # the mixer's reverb: the song's own when its header sets it (MPlayStart →
    # m4aSoundMode), else what m4aSoundInit left
    rv = (p.reverb & 0x7F) if p.reverb & 0x80 else m4a.reverb
    print(f'  {len(p.notes)} notes on {len({n.track for n in p.notes})} tracks '
          f'({sum(1 for n in p.notes if n.cgb)} PSG), {p.end_tick} ticks'
          + (f', loop {loop[0]}–{loop[1]}' if loop else ', no loop')
          + (f', {p.dropped} note(s) found no free channel' if p.dropped else ''))
    if args.solo_track is not None:
        p.notes = [n for n in p.notes if n.track == args.solo_track]
        p.curves = {k: v for k, v in p.curves.items() if k == args.solo_track}
    if not p.notes:
        print('  nothing to convert')
        return
    merged, variants, names = build_events(p, loop, title, m4a.master_volume, args.vibrato, warn)
    stem = os.path.splitext(out)[0]
    sf2 = stem + '.sf2'
    n = build_sf2(sf2, variants, rom, p.pcm_freq, title=title,
                  comment=f'{tags.get("game", "")} song {song}')
    print(f'  wrote {sf2} ({n} presets)')
    if m2t is None:
        m2t = load_midi2taud()
    m2t.parse_midi = lambda _p: (('ppq', PPQ), merged, names)
    argv = sys.argv
    grid = []
    if '--rpb' not in args.rest and '--speed' not in args.rest:
        rpb, speed = pick_grid(p)
        grid = ['--rpb', str(rpb), '--speed', str(speed)]
    sys.argv = (['midi2taud', stem + '.mid', sf2, out] + grid
                + (['-v'] if args.verbose else []) + args.rest)
    try:
        m2t.main()
    finally:
        sys.argv = argv
    if rv and args.echo and os.path.exists(out):
        # the taps are the PCM ring's slots P and P − 1 frames back (m4a_1.s
        # SoundMainRAM_Reverb; P = PCM_DMA_BUF_SIZE ÷ samples a frame): one
        # delay at their mean
        n = p.pcm_per_vblank
        P = 1584 // n
        sec = (P - ECHO_TAP) * n / p.pcm_freq
        gain = ECHO_GAIN * rv / 256.0
        r = subprocess.run(['node', os.path.join(HERE, 'addecho.mjs'), out, out, repr(gain), repr(sec)],
                           capture_output=True, text=True)
        print((r.stdout + r.stderr).rstrip() or '  (addecho printed nothing)')
        if r.returncode:
            warn('the echo could not be added; the .taud is without it')
    elif rv:
        warn(f'the mixer\'s reverb ({rv}) left out (--no-echo)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('output', nargs='?')
    ap.add_argument('--song', type=int, help='play this song number instead of the mini\'s')
    ap.add_argument('--list', action='store_true', help='list the song table')
    ap.add_argument('--seconds', type=float, default=900.0,
                    help='give up on a song that neither loops nor ends by then')
    ap.add_argument('--no-vibrato', dest='vibrato', action='store_false')
    ap.add_argument('--no-echo', dest='echo', action='store_false',
                    help='leave out the mixer\'s reverb (an IMix delay on the DirectSound instruments)')
    ap.add_argument('--solo-track', type=int, help=argparse.SUPPRESS)
    ap.add_argument('-v', '--verbose', action='store_true')
    argv = sys.argv[1:]
    rest = []
    if '--' in argv:
        i = argv.index('--')
        argv, rest = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    args.rest = rest

    if args.list:
        rom, tags, patches = load_gsf(args.input)
        m4a = M4A(rom)
        cur = song_number(patches)
        print(f'song table {m4a.song_table:#x}, players {m4a.mplay_table:#x}, mode {m4a.mode:#x}')
        i = 0
        while True:
            o = m4a.song_table - ROM_BASE + 8 * i
            if o + 8 > len(rom):
                break
            hdr = u32(rom, o)
            if hdr and not (ROM_BASE <= hdr < ROM_BASE + len(rom)):
                break
            if hdr and rom[hdr - ROM_BASE]:
                print(f"{'*' if i == cur else ' '}{i:5d}  header {hdr:#x}  tracks {rom[hdr - ROM_BASE]:2d}  "
                      f"player {struct.unpack_from('<H', rom, o + 4)[0]}")
            i += 1
            if i > 4096:
                break
        return

    if os.path.isdir(args.input):
        out_dir = args.output or args.input
        os.makedirs(out_dir, exist_ok=True)
        m2t = load_midi2taud()
        for f in sorted(os.listdir(args.input)):
            if f.lower().endswith(('.minigsf', '.gsf')):
                out = os.path.join(out_dir, os.path.splitext(f)[0] + '.taud')
                try:
                    convert(os.path.join(args.input, f), out, args, m2t)
                except SystemExit as e:
                    print(f'  {f}: {e}')
        return
    out = args.output or os.path.splitext(args.input)[0] + '.taud'
    convert(args.input, out, args)


if __name__ == '__main__':
    main()
