"""gbarom.py — GSF loading and the MP2K ("Sappy", m4a) song table.

A GSF is a PSF (version 0x22) whose program is [entry u32][load address u32]
[size u32][bytes]; `_lib` names a file loaded first, then the file's own
program is laid over it. Every rip here is the game's own ROM (the gsflib,
which boots the game's own m4a sound driver) and a 2-byte mini patch holding
the song number the patched game starts.
"""

import os
import struct
import zlib

ROM_BASE = 0x08000000


def _read_psf(path):
    b = open(path, 'rb').read()
    if b[:3] != b'PSF' or b[3] != 0x22:
        raise ValueError(f'{path}: not a GSF (PSF version 0x22)')
    rs, cs = struct.unpack_from('<II', b, 4)
    prog = zlib.decompress(b[16 + rs:16 + rs + cs]) if cs else b''
    tags = {}
    t = b[16 + rs + cs:]
    if t[:5] == b'[TAG]':
        for line in t[5:].decode('utf-8', 'replace').split('\n'):
            if '=' in line:
                k, v = line.split('=', 1)
                k = k.strip().lower()
                tags[k] = (tags[k] + '\n' + v.strip()) if k in tags else v.strip()
    return prog, tags


def load_gsf(path):
    """→ (rom bytearray indexed from 0x08000000, tags, patches) — `patches` lists
    the (ROM offset, bytes) the mini laid over its library."""
    prog, tags = _read_psf(path)
    rom = bytearray()
    patches = []

    def lay(p, record):
        if len(p) < 12:
            return
        _entry, addr, size = struct.unpack_from('<III', p, 0)
        data = p[12:12 + size]
        off = (addr & 0x01FFFFFF)
        if len(rom) < off + len(data):
            rom.extend(bytes(off + len(data) - len(rom)))
        rom[off:off + len(data)] = data
        if record:
            patches.append((off, bytes(data)))

    libs = [tags[k] for k in sorted(tags) if k == '_lib' or k.startswith('_lib')]
    for lib in libs:
        lp, _lt = _read_psf(os.path.join(os.path.dirname(path), lib))
        lay(lp, False)
    lay(prog, True)
    return rom, tags, patches


def u32(rom, off):
    return struct.unpack_from('<I', rom, off)[0] if 0 <= off <= len(rom) - 4 else 0


def is_ptr(rom, p):
    return ROM_BASE <= p < ROM_BASE + len(rom)


def _plausible_header(rom, p):
    if not is_ptr(rom, p) or p & 3:
        return False
    o = p - ROM_BASE
    if o + 8 > len(rom):
        return False
    n = rom[o]
    if n == 0:
        return True          # an empty song (a placeholder) is allowed in a table
    if n > 16 or not is_ptr(rom, u32(rom, o + 4)):
        return False
    return all(is_ptr(rom, u32(rom, o + 8 + 4 * i)) for i in range(n))


def find_song_table(rom, need=0):
    """The longest run of {header ptr, u16 ms, u16 me} whose headers look like
    SongHeaders ({trackCount, blockCount, priority, reverb, tone ptr, parts});
    it must reach entry `need`. → ROM offset of entry 0."""
    best = (0, None)
    off = 0
    n = len(rom) - 8
    while off < n:
        p = u32(rom, off)
        if _plausible_header(rom, p) and rom[p - ROM_BASE] > 0 and rom[off + 4 + 1] == 0 and rom[off + 6 + 1] == 0:
            run = 0
            o = off
            while o < n and _plausible_header(rom, u32(rom, o)) and rom[o + 5] == 0 and rom[o + 7] == 0 and rom[o + 4] < 32:
                run += 1
                o += 8
            if run > best[0] and run > need:
                # the entry before must not also be part of it (we want entry 0)
                best = (run, off)
            off = o if run > 4 else off + 4
        else:
            off += 4
    if best[1] is None:
        raise ValueError('no MP2K song table found')
    return best[1], best[0]


SONGNUM_START_SIG = bytes.fromhex('00B50004074A0849400B4018')


class M4A:
    """The game's m4a tables, found through m4aSongNumStart's code (whose
    literal pool names gMPlayTable and gSongTable) and m4aSoundInit's
    m4aSoundMode constant just before it (m4a.c)."""

    def __init__(self, rom):
        self.rom = rom
        h = rom.find(SONGNUM_START_SIG)
        if h < 0:
            raise ValueError('m4aSongNumStart not found: not an MP2K (m4a) game?')
        self.mplay_table = u32(rom, ((h + 4 + 4) & ~3) + 7 * 4)
        self.song_table = u32(rom, ((h + 6 + 4) & ~3) + 8 * 4)
        self.mode = None
        for o in range(h - 4, max(0, h - 0x200), -4):
            w = u32(rom, o)
            if ((w >> 20) & 0xFF) in (8, 9, 10, 11) and 1 <= (w >> 16) & 15 <= 12 \
                    and 1 <= (w >> 8) & 15 <= 12 and w >> 24 == 0:
                self.mode = w
                break
        if self.mode is None:
            self.mode = 0x0094C500          # the SDK's usual 13379 Hz, 5 channels, 12/15
        self.freq_index = (self.mode >> 16) & 15
        self.master_volume = (self.mode >> 12) & 15
        self.max_chans = (self.mode >> 8) & 15
        self.reverb = (self.mode & 0x7F) if self.mode & 0x80 else 0

    def song(self, n):
        """→ (header address, music player index, player's track count)."""
        o = self.song_table - ROM_BASE + 8 * n
        hdr = u32(self.rom, o)
        ms = struct.unpack_from('<H', self.rom, o + 4)[0]
        po = self.mplay_table - ROM_BASE + 12 * ms
        ntr = self.rom[po + 8]
        return hdr, ms, ntr


def song_number(patches):
    for off, data in patches:
        if len(data) == 2:
            return struct.unpack('<H', data)[0]
    raise ValueError('the mini carries no 2-byte song number')


if __name__ == '__main__':
    import sys
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == '--dump-rom':
        # gbarom.py --dump-rom in.minigsf out.gba [--solo N]: the ROM image a
        # GSF player runs; --solo zeroes every VOL argument of the song's other
        # tracks (found by running the song), for comparing one track at a time
        rom, tags, patches = load_gsf(a[1])
        if '--solo' in a:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from m4aplay import simulate
            solo = int(a[a.index('--solo') + 1])
            m = M4A(rom)
            hdr, _ms, ntr = m.song(song_number(patches))
            p, _loop = simulate(bytes(rom), m, hdr, ntr, max_seconds=600, warn=lambda _m: None)
            keep = p.vol_args.get(solo, set())        # patterns can be shared between tracks
            for t, addrs in p.vol_args.items():
                if t != solo:
                    for ad in addrs - keep:
                        rom[ad - ROM_BASE] = 0
        open(a[2], 'wb').write(rom)
        print(len(rom))
