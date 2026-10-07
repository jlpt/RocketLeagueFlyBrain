#!/usr/bin/env python3
"""BWV 542 x U.N. Owen Was Her? -- one new melody fused from both tunes.

Instead of playing the two tunes on top of each other, the arrangement is
built on new melodies whose phrases are stitched together from J. S. Bach's
fugue subject (Fantasia and Fugue in G minor, BWV 542) and ZUN's
"U.N. Owen Was Her?" (Touhou 6), so a single line keeps turning from one
into the other.

The seams
---------
Everything is in G minor (Owen is moved down from D minor) at Owen's
155 bpm; Bach's fragments run at half time, so his sixteenths become
eighths and move at the same speed as Owen's line.

* Theme A ("verse")  -- Bach's opening turn  D | Bb C A Bb G G'  ends on G,
  which is exactly the note Owen's theme starts on, so the line carries on
  with Owen's bars 2-4.  The answering phrase starts with Owen's bar 1, puts
  Bach's  F# G E F# D G D  where Owen's bar 2 was, and ends with Owen.
* Theme B ("chorus") -- Owen's falling-third chromatic line (D Bb C# A |
  E C# C Eb) runs into Bach's falling-third sequence (D Eb C D Bb Eb Bb |
  C A D D'), twice, and closes with Bach's turn into a cadence on G.
* Theme C ("bridge") -- Bach's rising run (Bb C A Bb G A Bb C | D E F G F G)
  climbs into Owen's bars 1-3 and hands over to Bach's sequence, ending on
  the dominant.
* Riff -- Owen's 5/4 intro arpeggio, where every other stab is replaced by
  Bach's turn, as one harpsichord line.

Sources (passed on the command line, not bundled):
  * BWV 542 fugue MIDI from the Mutopia Project edition (public domain):
    https://www.mutopiaproject.org/ftp/BachJS/BWV542/bwv542/bwv542-mids.zip
    -> bwv542-a4-1.mid (used to verify the Bach fragments and for the final cadence)
  * ZUN's SC-88Pro MIDI of U.N. Owen Was Her? (used only to verify the Owen
    fragments; melodies, harmony and orchestration are written here).

Usage:
  python3 make_mashup.py FUGUE.mid UN_OWEN_ZUN.mid OUT.mid
"""
import random
import sys

import mido

TPB = 480
Q = TPB
S16 = TPB // 4
BB = 2 * Q            # one Bach beat (the fugue runs at half time)
BAR = 16 * S16

_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def P(name):
    """'Bb4' -> 70 (MIDI note number, C4 = 60)."""
    i, acc = 1, 0
    while name[i] in "#b":
        acc += 1 if name[i] == "#" else -1
        i += 1
    return 12 * (int(name[i:]) + 1) + _STEP[name[0]] + acc


def harsh(a, b):
    """minor 2nd / minor 9th (a major 7th is left alone: it is a normal voicing)"""
    lo, hi = sorted((a, b))
    return (hi - lo) % 12 == 1


# ---------------------------------------------------------------- the melodies
# Each bar: list of (16th position, length in 16ths, note); G minor, written pitch.
# Comments say where each bar comes from.
THEME_A = dict(
    pickup=[(12, 4, "D5")],                                                   # Bach up-beat
    bars=[
        [(0, 2, "Bb4"), (2, 2, "C5"), (4, 2, "A4"), (6, 2, "Bb4"), (8, 3, "G4"), (12, 3, "G5")],  # Bach
        [(0, 2, "Bb5"), (4, 2, "C6"), (6, 2, "D6"), (8, 2, "C6"), (12, 2, "E6")],                 # Owen 2
        [(0, 2, "G6"), (2, 2, "D6"), (4, 2, "A6"), (6, 1, "Bb6"), (8, 2, "A6"), (10, 1, "Bb6"),
         (11, 1, "A6"), (12, 2, "G6"), (14, 1, "F6")],                                             # Owen 3
        [(0, 2, "D6"), (2, 2, "F6"), (4, 2, "C6"), (6, 2, "D6"), (8, 8, "Bb5")],                  # Owen 4
        [(0, 2, "G5"), (4, 2, "D6"), (8, 2, "A5"), (12, 2, "D6")],                                # Owen 1
        [(0, 2, "F#6"), (2, 2, "G6"), (4, 2, "E6"), (6, 2, "F#6"), (8, 4, "D6"), (12, 2, "G6"),
         (14, 2, "D6")],                                                                           # Bach
        [(0, 2, "G6"), (2, 2, "D6"), (4, 2, "A6"), (6, 1, "Bb6"), (8, 2, "A6"), (10, 1, "Bb6"),
         (11, 1, "A6"), (12, 2, "G6"), (14, 1, "F6")],                                             # Owen 3
        [(0, 12, "G6")],                                                                           # Owen end
    ],
    chords=["Gm Gm", "Gm C/E", "Eb F", "Gm Gm", "Eb F", "D7 Gm", "Eb F", "Gm Gm"],
)
THEME_B = dict(
    bars=[
        [(0, 2, "D6"), (4, 2, "Bb5"), (8, 2, "C#6"), (12, 2, "A5")],                              # Owen chromatic
        [(0, 2, "E6"), (4, 2, "C#6"), (8, 2, "C6"), (12, 2, "Eb6")],                              # Owen chromatic
        [(0, 2, "D6"), (2, 2, "Eb6"), (4, 2, "C6"), (6, 2, "D6"), (8, 4, "Bb5"), (12, 2, "Eb6"),
         (14, 2, "Bb5")],                                                                          # Bach sequence
        [(0, 3, "C6"), (4, 3, "A5"), (8, 3, "D5"), (12, 3, "D6")],                                # Bach sequence
        [(0, 2, "D6"), (4, 2, "Bb5"), (8, 2, "C#6"), (12, 2, "A5")],                              # Owen chromatic
        [(0, 2, "E6"), (4, 2, "C#6"), (8, 2, "C6"), (12, 2, "Eb6")],                              # Owen chromatic
        [(0, 2, "Bb5"), (2, 2, "C6"), (4, 2, "A5"), (6, 2, "Bb5"), (8, 2, "A5"), (10, 2, "G5"),
         (12, 4, "F#5")],                                                                          # Bach turn + cadence
        [(0, 8, "G5")],
    ],
    chords=["Gm F#m", "A Ab", "Bb Eb", "Am7b5 D7", "Gm F#m", "A Ab", "Gm D7", "Gm Gm"],
)
THEME_C = dict(
    bars=[
        [(0, 2, "Bb4"), (2, 2, "C5"), (4, 2, "A4"), (6, 2, "Bb4"), (8, 2, "G4"), (10, 2, "A4"),
         (12, 2, "Bb4"), (14, 2, "C5")],                                                           # Bach run
        [(0, 4, "D5"), (4, 4, "E5"), (8, 2, "F5"), (10, 2, "G5"), (12, 2, "F5"), (14, 2, "G5")],  # Bach run
        [(0, 2, "G5"), (4, 2, "D6"), (8, 2, "A5"), (12, 2, "D6")],                                # Owen 1
        [(0, 2, "Bb5"), (4, 2, "C6"), (6, 2, "D6"), (8, 2, "C6"), (12, 2, "E6")],                 # Owen 2
        [(0, 2, "G6"), (2, 2, "D6"), (4, 2, "A6"), (6, 1, "Bb6"), (8, 2, "A6"), (10, 1, "Bb6"),
         (11, 1, "A6"), (12, 2, "G6"), (14, 1, "F6")],                                             # Owen 3
        [(0, 3, "Eb6"), (4, 3, "C6"), (8, 3, "F5"), (12, 3, "F6")],                               # Bach sequence
        [(0, 2, "D6"), (2, 2, "Eb6"), (4, 2, "C6"), (6, 2, "D6"), (8, 4, "Bb5"), (12, 2, "Eb6"),
         (14, 2, "Bb5")],                                                                          # Bach sequence
        [(0, 3, "C6"), (4, 3, "A5"), (8, 3, "D5"), (12, 3, "D6")],                                # Bach sequence
    ],
    chords=["Gm Gm", "C F", "Eb F", "Gm C/E", "Eb F", "Cm F", "Bb Eb", "Am7b5 D7"],
)

# chord name -> (bass root, voicing)
CHORDS = {
    "Gm": ("G", ["G3", "Bb3", "D4"]), "G": ("G", ["G3", "B3", "D4"]),
    "C/E": ("E", ["E3", "G3", "C4"]), "C": ("C", ["C3", "E3", "G3"]), "Cm": ("C", ["Eb3", "G3", "C4"]),
    "Eb": ("Eb", ["Eb3", "G3", "Bb3"]), "F": ("F", ["F3", "A3", "C4"]), "Bb": ("Bb", ["Bb3", "D4", "F4"]),
    "D": ("D", ["D3", "F#3", "A3"]), "D7": ("D", ["D3", "F#3", "A3", "C4"]),
    "F#m": ("F#", ["F#3", "A3", "C#4"]), "A": ("A", ["A3", "C#4", "E4"]), "Ab": ("Ab", ["Ab3", "C4", "Eb4"]),
    "Am7b5": ("A", ["A3", "C4", "Eb4", "G4"]),
}

# the real fragments, used to check the hand-written themes against the sources
SUBJECT = [
    (1.5, .5, "D5"),
    (2, .25, "Bb4"), (2.25, .25, "C5"), (2.5, .25, "A4"), (2.75, .25, "Bb4"),
    (3, .5, "G4"), (3.5, .5, "G5"),
    (4, .25, "F#5"), (4.25, .25, "G5"), (4.5, .25, "E5"), (4.75, .25, "F#5"),
    (5, .5, "D5"), (5.5, .25, "G5"), (5.75, .25, "D5"),
    (6, .5, "Eb5"), (6.5, .5, "C5"), (7, .5, "F4"), (7.5, .5, "F5"),
    (8, .25, "D5"), (8.25, .25, "Eb5"), (8.5, .25, "C5"), (8.75, .25, "D5"),
    (9, .5, "Bb4"), (9.5, .25, "Eb5"), (9.75, .25, "Bb4"),
    (10, .5, "C5"), (10.5, .5, "A4"), (11, .5, "D4"), (11.5, .5, "D5"),
    (12, .25, "Bb4"), (12.25, .25, "C5"), (12.5, .25, "A4"), (12.75, .25, "Bb4"),
    (13, .25, "G4"), (13.25, .25, "A4"), (13.5, .25, "Bb4"), (13.75, .25, "C5"),
    (14, .5, "D5"), (14.5, .5, "E5"),
    (15, .25, "F5"), (15.25, .25, "G5"), (15.5, .25, "F5"), (15.75, .25, "G5"),
]


# ---------------------------------------------------------------- reading
def read_notes(path):
    m = mido.MidiFile(path)
    tpb = m.ticks_per_beat
    out = []
    for ti, track in enumerate(m.tracks):
        tick, on = 0, {}
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                on.setdefault((msg.channel, msg.note), []).append(tick)
            elif msg.type in ("note_off", "note_on") and on.get((msg.channel, msg.note)):
                s = on[(msg.channel, msg.note)].pop(0)
                out.append(dict(start=s / tpb, dur=(tick - s) / tpb, pitch=msg.note,
                                track=ti, ch=msg.channel))
    out.sort(key=lambda n: (n["start"], n["pitch"]))
    return out


def owen_top(notes, ch, a, b, transpose):
    """top voice of ZUN's channel `ch` in the 16th-grid window [a, b): {rel16: pitch}"""
    out = {}
    for n in notes:
        q = round(n["start"] * 4)
        if n["ch"] == ch and a <= q < b:
            out[q - a] = max(out.get(q - a, 0), n["pitch"] + transpose)
    return out


def verify_sources(FU, OW):
    """Every Bach / Owen bar in the themes must be a literal fragment of the source."""
    top = {}
    for n in FU:
        if n["track"] == 1 and n["start"] < 16:
            top[n["start"]] = max(top.get(n["start"], 0), n["pitch"])
    for b, d, nm in SUBJECT:
        assert top.get(b) == P(nm), ("Bach subject", b, nm)
    bach = {b: P(nm) for b, d, nm in SUBJECT}            # Bach beat -> pitch

    def bach_bar(bar, b0, octave=0):                      # Bach at half time: 1 beat = 8 sixteenths
        for pos, d, nm in bar:
            assert bach[b0 + pos / 8] + octave == P(nm), ("Bach", b0, pos, nm)

    owen = owen_top(OW, 6, 312, 440, -7)                  # main theme A + A' in G minor
    chrom = owen_top(OW, 6, 248, 312, +5)                 # chromatic theme in G minor

    def owen_bar(bar, src, u0):
        for pos, d, nm in bar:
            assert src.get(u0 + pos) == P(nm), ("Owen", u0, pos, nm)

    a, b, c = THEME_A["bars"], THEME_B["bars"], THEME_C["bars"]
    bach_bar(a[0], 2), bach_bar(a[5], 4, 12)
    owen_bar(a[1], owen, 16), owen_bar(a[2], owen, 32), owen_bar(a[3], owen, 48)
    owen_bar(a[4], owen, 0), owen_bar(a[6], owen, 96)
    owen_bar(b[0], chrom, 0), owen_bar(b[1], chrom, 16)
    bach_bar(b[2], 8, 12), bach_bar(b[3], 10, 12), bach_bar(b[6][:4], 12, 12)
    bach_bar(c[0], 12), bach_bar(c[1], 14)
    owen_bar(c[2], owen, 0), owen_bar(c[3], owen, 16), owen_bar(c[4], owen, 32)
    bach_bar(c[5], 6, 12), bach_bar(c[6], 8, 12), bach_bar(c[7], 10, 12)


# ---------------------------------------------------------------- writing
CHANNELS = {
    # name: (channel, GM program, track name, volume, pan, reverb, chorus)
    "organ": (0, 19, "Church organ - manuals", 96, 60, 92, 0),
    "pedal": (1, 19, "Church organ - pedal", 100, 64, 88, 0),
    "lead":  (2, 80, "Square lead", 92, 66, 55, 30),
    "dbar":  (3, 16, "Drawbar organ", 72, 72, 55, 20),
    "hpsi":  (4, 6, "Harpsichord", 84, 36, 60, 0),
    "str":   (5, 48, "Strings", 74, 92, 85, 40),
    "bass":  (6, 33, "Finger bass", 96, 64, 25, 0),
    "cele":  (7, 8, "Celesta", 92, 96, 75, 0),
    "choir": (8, 52, "Choir aahs", 76, 64, 105, 30),
    "drums": (9, 16, "Drums (power kit)", 88, 64, 45, 0),
    "timp":  (10, 47, "Timpani", 96, 64, 80, 0),
    "gtr":   (11, 29, "Overdrive guitar", 66, 30, 45, 0),
    "brass": (12, 61, "Brass section", 72, 80, 70, 10),
    "glock": (13, 9, "Glockenspiel", 90, 100, 70, 0),
}

K, K2, RS, SN, CP, CH, PH, OH = 36, 35, 37, 38, 39, 42, 44, 46
CR, CR2, RD, SPL, CHN = 49, 57, 51, 55, 52
TLO, TMID, THI = 45, 47, 50


class Arrangement:
    def __init__(self, seed=542):
        self.notes = []   # (name, tick, dur, pitch, vel)
        self.ccs = []     # (name, tick, cc, value)
        self.meta = []    # (tick, kind, kwargs)
        self.rnd = random.Random(seed)

    def note(self, name, tick, dur, pitch, vel, human=4):
        tick, dur = int(round(tick)), int(round(dur))
        if dur <= 0 or not 0 <= pitch <= 127:
            return
        if human:
            vel += self.rnd.randint(-human, human)
        self.notes.append((name, tick, dur, int(pitch), max(1, min(127, int(vel)))))

    def chord(self, name, tick, dur, pitches, vel, human=3):
        for p in pitches:
            self.note(name, tick, dur, p, vel, human)

    def cc(self, name, tick, ctl, val):
        self.ccs.append((name, int(round(tick)), ctl, max(0, min(127, int(val)))))

    def ramp(self, name, t0, t1, ctl, v0, v1, steps=16):
        for i in range(steps + 1):
            self.cc(name, t0 + (t1 - t0) * i / steps, ctl, v0 + (v1 - v0) * i / steps)

    def mix(self, tick, **vols):
        for name, v in vols.items():
            self.cc(name, tick, 7, v)

    def tempo(self, tick, bpm):
        self.meta.append((int(tick), "set_tempo", dict(tempo=mido.bpm2tempo(bpm))))

    def timesig(self, tick, num, den=4):
        self.meta.append((int(tick), "time_signature", dict(numerator=num, denominator=den)))

    def marker(self, tick, text):
        self.meta.append((int(tick), "marker", dict(text=text)))

    def drums(self, t0, patterns, scale=1.0, step=S16):
        """patterns: {drum note: 'X.x.o...'}; X=accent, x=normal, o=ghost."""
        vmap = {"X": 112, "x": 88, "o": 58}
        for drum, pat in patterns.items():
            for i, c in enumerate(pat):
                if c in vmap:
                    self.note("drums", t0 + i * step, S16, drum, vmap[c] * scale, human=5)

    def timp_roll(self, t0, t1, pitch, v0, v1, rate=S16 // 2):
        n = max(1, int((t1 - t0) // rate))
        for i in range(n):
            self.note("timp", t0 + i * rate, rate, pitch, v0 + (v1 - v0) * i / max(1, n - 1), human=3)

    def save(self, path, title):
        by_key = {}
        for n in self.notes:
            by_key.setdefault((n[0], n[3]), []).append(list(n))
        cleaned = []
        for lst in by_key.values():
            lst.sort(key=lambda n: (n[1], -n[4]))       # louder first on equal onsets
            kept = []
            for n in lst:
                if kept and kept[-1][1] == n[1]:          # duplicate onset: merge into louder
                    kept[-1][2] = max(kept[-1][2], n[2])
                    continue
                kept.append(n)
            for a, b in zip(kept, kept[1:]):             # re-articulate overlapping repeats
                if a[1] + a[2] > b[1] - 8:
                    a[2] = max(S16 // 4, b[1] - 8 - a[1])
            cleaned.extend(kept)
        mf = mido.MidiFile(type=1, ticks_per_beat=TPB)
        cond = mido.MidiTrack()
        mf.tracks.append(cond)
        evs = [(0, 0, mido.MetaMessage("track_name", name=title, time=0))]
        for tick, kind, kw in self.meta:
            evs.append((tick, 0, mido.MetaMessage(kind, time=0, **kw)))
        evs.append((max(n[1] + n[2] for n in cleaned) + TPB * 2, 9,
                    mido.MetaMessage("end_of_track", time=0)))
        _flush(cond, evs)
        for name, (ch, prog, tname, vol, pan, rev, cho) in CHANNELS.items():
            tr = mido.MidiTrack()
            mf.tracks.append(tr)
            ev = [(0, 0, mido.MetaMessage("track_name", name=tname, time=0))]
            if ch == 9:
                ev.append((0, 1, mido.Message("control_change", channel=ch, control=0, value=0)))
            ev.append((0, 2, mido.Message("program_change", channel=ch, program=prog)))
            for ctl, val in ((7, vol), (10, pan), (91, rev), (93, cho), (11, 127), (64, 0)):
                ev.append((0, 3, mido.Message("control_change", channel=ch, control=ctl, value=val)))
            for nm, tick, ctl, val in self.ccs:
                if nm == name:
                    ev.append((tick, 4, mido.Message("control_change", channel=ch, control=ctl, value=val)))
            for nm, tick, dur, pitch, vel in cleaned:
                if nm == name:
                    ev.append((tick, 6, mido.Message("note_on", channel=ch, note=pitch, velocity=vel)))
                    ev.append((tick + dur, 5, mido.Message("note_off", channel=ch, note=pitch, velocity=0)))
            _flush(tr, ev)
        mf.save(path)
        return mf


def _flush(track, evs):
    evs.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for tick, _, msg in evs:
        msg.time = tick - last
        last = tick
        track.append(msg)
    if not (track and track[-1].type == "end_of_track"):
        track.append(mido.MetaMessage("end_of_track", time=0))


MAIN_GROOVE = {K: "X...X...X...X...", CH: "x...x...x...x...", OH: "..x...x...x...x.",
               CP: "....x.......x..."}


# ================================================================ sections
def fused_riff(A, T0, cycles, full, last_chord="D7"):
    """Owen's 5/4 intro arpeggio with every other stab replaced by Bach's turn
    (Bb C A Bb G G'), as one harpsichord line.  The last stab is `last_chord`,
    which sets up the next theme's up-beat."""
    CYC = 20
    arp = [P("Bb4"), P("G4"), P("D4")]
    head = [(12, 1, "Bb4"), (13, 1, "C5"), (14, 1, "A4"), (15, 1, "Bb4"), (16, 2, "G4"), (18, 2, "G5")]
    stab = [P("Eb4"), P("A4"), P("C5")]
    last = {"D7": ([P("C4"), P("F#4"), P("A4")], P("D3")), "A7": ([P("C#4"), P("E4"), P("G4")], P("A2"))}
    for k in range(cycles):
        t = T0 + k * CYC * S16
        bach_cycle = k % 2 == 0
        line_v = 84 if full or k >= 2 else 78
        for i in range(12):                                         # Owen's arpeggio
            A.note("hpsi", t + i * S16, S16, arp[i % 3], line_v - (6 if i % 3 else 0))
            if full or k >= 2:
                A.note("cele", t + i * S16, S16, arp[i % 3] + 12, 64 + (6 if full else 2 * k))
        if bach_cycle:                                              # Bach's turn closes the bar
            for pos, d, nm in head:
                A.note("hpsi", t + pos * S16, d * S16, P(nm), 96)
                if full or k >= 2:
                    A.note("cele", t + pos * S16, d * S16, P(nm) + 12, 76)
                if full or k >= 4:
                    A.note("organ", t + pos * S16, d * S16 - 10, P(nm), 86)
            bass = [(0, 6, "G2"), (6, 4, "D3"), (10, 2, "G3"), (12, 4, "G2"), (16, 4, "Bb2")]
        else:                                                       # Owen's stabs
            for j, pos in enumerate((12, 14, 17)):
                ch = stab
                if k == cycles - 1 and pos == 17:
                    ch = last[last_chord][0]
                d = 2 if pos == 17 else 1
                A.chord("hpsi", t + pos * S16, d * S16 + 30, ch, 100)
                if full or k >= 3:
                    A.chord("organ", t + pos * S16, d * S16 + 20, ch, 84)
                    A.chord("str", t + pos * S16, d * S16 + 20, ch, 74)
                if full or k >= 5:
                    A.chord("brass", t + pos * S16, d * S16 + 20, [p + 12 for p in ch], 80)
            bass = [(0, 6, "G2"), (6, 4, "D3"), (10, 2, "G3"), (12, 4, "C3"), (16, 4, "C3")]
        for pos, d, nm in bass:
            p = P(nm)
            if k == cycles - 1 and pos == 16:
                p = last[last_chord][1]
            if full or k >= 2:
                A.note("bass", t + pos * S16, d * S16 - 30, p, 94)
            if full or k >= 4:
                A.note("pedal", t + pos * S16, d * S16 - 30, p - 12, 72)
        if k == cycles - 1:
            A.drums(t, {K: "X.....X.....X.X.....", CH: "x.x.x.x.x.x.........",
                        SN: "....x.....x.oxxxXXXX", TMID: "............x......."})
        elif full or k >= 4:
            A.drums(t, {K: "X.....X.....X.X..X..", SN: "....x.....x......X..", CH: "x.x.x.x.x.x.........",
                        OH: "............x.......", CR: "X" if k % 4 == 0 else ""})
        elif k >= 2:
            A.drums(t, {RD: "x...x...x...x.......", K: "X...........x.x..x..", PH: "..o...o...o........."}, 0.85)
    if not full:
        A.note("pedal", T0, 4 * CYC * S16 - 40, P("G1"), 74)
        A.note("pedal", T0, 4 * CYC * S16 - 40, P("G2"), 56)


def theme(A, T, th, tr, opt):
    """Play one of the fused themes (8 bars from tick T, up-beat before it),
    transposed by `tr`, with accompaniment chosen by `opt`."""
    # chord timeline (half bars), consecutive equal chords merged
    halves = []
    for i, pair in enumerate(th["chords"]):
        for j, name in enumerate(pair.split()):
            halves.append([T + i * BAR + j * 8 * S16, T + i * BAR + (j + 1) * 8 * S16, name])
    if opt.get("last"):
        halves[-1][2] = opt["last"]
    spans = []
    for h in halves:
        if spans and spans[-1][2] == h[2]:
            spans[-1][1] = h[1]
        else:
            spans.append(list(h))

    def chord_at(t):
        for t0, t1, name in halves:
            if t0 <= t < t1:
                return name
        return halves[0][2] if t < T else halves[-1][2]

    def pcs(name):
        return {(P(v) + tr) % 12 for v in CHORDS[name][1]}

    # melody (+ a harmony voice a 3rd/4th below, chord tones only)
    notes = [(T - BAR + pos * S16, d, P(nm) + tr) for pos, d, nm in th.get("pickup", [])]
    for i, bar in enumerate(th["bars"]):
        notes += [(T + i * BAR + pos * S16, d, P(nm) + tr) for pos, d, nm in bar]
    mel = []
    for t, d, p in notes:
        dur = d * S16 - (18 if d <= 3 else 0)
        for inst, octv, vel in opt["lead"]:
            A.note(inst, t, dur, p + octv, vel)
            mel.append((t, t + dur, p + octv))
        if opt.get("harm") and d >= 2 and t >= T:            # no harmony on the up-beat
            cands = [h for h in range(p - 5, p - 2) if h % 12 in pcs(chord_at(t))]
            if cands:
                for inst, octv, vel in opt["harm"]:
                    A.note(inst, t, dur, max(cands) + octv, vel)
                    mel.append((t, t + dur, max(cands) + octv))

    def clashes(t0, t1, pitch):
        return any(s < t1 and e > t0 and harsh(pitch, m) for s, e, m in mel)

    # harmony
    for t0, t1, name in spans:
        root, voicing = CHORDS[name]
        bp = P(root + "2") + tr
        while bp > P("Ab2"):
            bp -= 12
        while bp < P("A1"):
            bp += 12
        padp = [P(v) + tr for v in voicing]
        while min(padp) < P("D3"):
            padp = [x + 12 for x in padp]
        while min(padp) > P("C#4"):
            padp = [x - 12 for x in padp]
        for inst, octv, vel in opt.get("pad", []):
            keep = [x + octv for x in padp if not clashes(t0, t1, x + octv)]
            A.chord(inst, t0, t1 - t0 - 15, keep or [padp[0] + octv], vel)
        steps = (t1 - t0) // (2 * S16)
        if opt.get("bass") == "roots8":
            for e in range(steps):
                A.note("bass", t0 + e * 2 * S16, 2 * S16 - 30, bp + (12 if e % 2 else 0), 96 if e % 2 == 0 else 84)
        elif opt.get("bass") == "long":
            A.note("bass", t0, t1 - t0 - 30, bp, 86)
        if opt.get("pedal") == "long":
            A.note("pedal", t0, t1 - t0 - 20, bp, 82)
            A.note("pedal", t0, t1 - t0 - 20, bp + 12, 62)
        elif opt.get("pedal") == "walk":                       # baroque octave walk in quarters
            for e in range((t1 - t0) // Q):
                A.note("pedal", t0 + e * Q, Q - 40, bp + (12 if e % 2 else 0), 88)
        if opt.get("gtr"):
            fifth = bp + 7 if (bp + 7) % 12 in pcs(name) else bp + 12
            for e in range(steps):
                A.chord("gtr", t0 + e * 2 * S16, int(1.5 * S16), [bp + 12, fifth + 12], 80 if e % 2 == 0 else 66)
        if opt.get("arps"):                                     # Owen-style 3-note broken chords
            inst, vel = opt["arps"]
            tones = sorted({P("G3") + ((p - P("G3")) % 12) for p in padp})[-3:][::-1]   # below the tune
            for i in range((t1 - t0) // S16):
                tt = t0 + i * S16
                for cand in tones[i % 3:] + tones[:i % 3]:
                    if not clashes(tt, tt + S16, cand):
                        A.note(inst, tt, S16, cand, vel - (6 if i % 3 else 0))
                        break
        if opt.get("chop"):                                     # continuo-style chords on the beat
            inst, vel = opt["chop"]
            for e in range((t1 - t0) // Q):
                keep = [x + 12 for x in padp if not clashes(t0 + e * Q, t0 + e * Q + Q // 2, x + 12)]
                A.chord(inst, t0 + e * Q, Q // 2, keep, vel)
    if opt.get("timp"):
        for i in range(0, 8, 2):
            name = chord_at(T + i * BAR)
            bp = P(CHORDS[name][0] + "2") + tr
            while bp > P("D3"):
                bp -= 12
            while bp < P("F2"):
                bp += 12
            A.note("timp", T + i * BAR, Q, bp, 94)
    # drums
    style = opt.get("drums")
    for bar in range(8):
        t = T + bar * BAR
        if style == "light":
            pat = {CH: "o.x.o.x.o.x.o.x.", K: "x.......x......."}
            if bar == 0:
                pat[CR] = "x"
        elif style in ("groove", "big", "build"):
            pat = dict(MAIN_GROOVE)
            if style == "big":
                pat[CH] = "x.o.x.o.x.o.x.o."
            if style == "build" and bar >= 4:
                pat[SN] = "....x.......x.x." if bar < 6 else "..x.x.x.x.x.x.x."
            if bar in ((0, 4) if style != "big" else (0, 2, 4, 6)):
                pat[CR] = "X"
        else:
            continue
        if bar == 7:
            pat.pop(CP, None)
            pat[SN] = {"light": "............o.xx", "groove": "..........x.xxXx",
                       "big": "........x.xxXXXX", "build": "xxxxxxxxXXXXXXXX"}[style]
        A.drums(t, pat, 0.9 if style == "light" else 1.0)


def cadence(A, FU, T0):
    """Bach's own final bar (m114) and G-major chord, with Owen's arpeggio on top."""
    def ct(beat):
        return T0 + (beat - 452) * BB
    hold = 4 * BB
    for n in FU:
        if n["track"] not in (1, 3, 5) or not 452 <= n["start"] < 457:
            continue
        s, d, p = n["start"], n["dur"], n["pitch"]
        dd = (hold if s >= 456 else d * BB) - 24
        if n["track"] == 5:
            A.note("pedal", ct(s), dd, p, 94)
            A.note("pedal", ct(s), dd, p - 12, 78)
            A.note("bass", ct(s), dd, p, 96)
        else:
            A.note("organ", ct(s), dd, p, 96, human=2)
            A.note("choir", ct(s), dd, p, 80)
            A.note("str", ct(s), dd, p + (12 if n["track"] == 3 else 0), 78)
    tf = ct(456)
    A.chord("brass", tf, hold, [P("G3"), P("D4"), P("G4"), P("B4"), P("D5")], 92)
    A.chord("lead", tf, hold - Q, [P("B5"), P("D6"), P("G6")], 90)
    A.chord("dbar", tf, hold - Q, [P("G4"), P("B4"), P("D5")], 76)
    A.chord("gtr", tf, hold - Q, [P("G2"), P("D3"), P("G3")], 92)
    A.note("bass", tf, hold, P("G1"), 100)
    for i, nm in enumerate(["B5", "G5", "D5"] * 4):        # Owen's arpeggio, now in G major
        A.note("cele", tf + i * S16, 3 * S16, P(nm), 86 - 3 * i)
        A.note("glock", tf + i * S16, 2 * S16, P(nm), 74 - 3 * i)
    A.drums(tf, {K: "X", CR: "X", CR2: "X", CHN: "X"})
    A.timp_roll(tf, tf + hold - Q, P("G2"), 116, 40, rate=Q // 8)
    A.ramp("choir", tf, tf + hold, 11, 127, 70)
    A.ramp("str", tf, tf + hold, 11, 127, 60)
    A.drums(T0, {K: "X.......X.......X.......X.......", SN: "........X...............X.......",
                 CR: "X", TLO: "..........................x.xxxx",
                 TMID: "........................xx.x...."})
    for beat, bpm in ((452, 150), (453, 142), (454, 130), (455, 112), (455.5, 94), (456, 66)):
        A.tempo(ct(beat), bpm)
    return tf + hold


# ================================================================ build
def build(fugue_path, owen_path, out_path):
    A = Arrangement()
    FU = read_notes(fugue_path)
    OW = read_notes(owen_path)
    verify_sources(FU, OW)

    T = 0
    A.tempo(0, 155)

    def riff(label, cycles, full, last_chord):
        nonlocal T
        A.marker(T, label)
        A.timesig(T, 5)
        fused_riff(A, T, cycles, full, last_chord)
        T += cycles * 20 * S16
        A.timesig(T, 4)

    def play(label, th, tr=0, **opt):
        nonlocal T
        A.marker(T, label)
        theme(A, T, th, tr, opt)
        T += 8 * BAR

    A.mix(0, drums=72)
    riff("Intro: fused riff (Owen's arpeggio + Bach's turn)", 8, False, "D7")
    A.mix(T, drums=70)
    play("Theme A (verse), music box", THEME_A,
         lead=[("cele", 0, 94), ("glock", 0, 84)], pad=[("str", 0, 58)], bass="long",
         arps=("hpsi", 56), drums="light")
    A.mix(T, drums=72, lead=100, organ=84)
    play("Theme A (verse), band", THEME_A,
         lead=[("lead", 0, 100), ("dbar", 0, 80)], harm=[("lead", 0, 76), ("dbar", 0, 62)],
         pad=[("organ", 0, 70)], bass="roots8", arps=("hpsi", 60), drums="groove")
    A.mix(T, drums=76, lead=100, organ=96)
    play("Theme B (chorus)", THEME_B,
         lead=[("lead", 0, 102), ("organ", -12, 88)], harm=[("lead", 0, 78)],
         pad=[("str", 0, 66), ("choir", 0, 60)], bass="roots8", drums="groove", timp=True)
    A.mix(T, drums=74, organ=96, hpsi=100)
    riff("Fused riff interlude", 4, True, "A7")
    A.mix(T, organ=104)
    play("Theme A in D minor, organ (baroque episode)", THEME_A, -5, last="G",
         lead=[("organ", 0, 98)], chop=("hpsi", 70), pad=[("str", 0, 50)], pedal="walk")
    A.mix(T, drums=74, lead=100, organ=90)
    A.ramp("str", T, T + 8 * BAR, 11, 80, 127)
    play("Theme C (bridge), building", THEME_C,
         lead=[("lead", 0, 100), ("glock", 0, 70)], harm=[("dbar", 0, 66)],
         pad=[("organ", 0, 66), ("str", 0, 62)], bass="roots8", drums="build")
    A.timp_roll(T - 2 * Q, T, P("D3"), 50, 110)
    A.mix(T, drums=80, lead=104, organ=100)
    A.ramp("choir", T, T + 2 * BAR, 11, 70, 120)
    play("Theme B (chorus), climax", THEME_B, last="D7",
         lead=[("lead", 0, 104), ("organ", -12, 92), ("brass", -12, 70)], harm=[("lead", 0, 80), ("dbar", 0, 66)],
         pad=[("choir", 0, 72), ("str", 12, 64)], bass="roots8", pedal="long", gtr=True, drums="big", timp=True)
    A.mix(T, drums=80, lead=104, organ=100)
    play("Theme A (verse), finale", THEME_A,
         lead=[("lead", 0, 106), ("organ", -12, 94), ("glock", 0, 64)], harm=[("lead", 0, 82), ("dbar", 0, 68)],
         pad=[("choir", 0, 74), ("str", 12, 66), ("brass", 0, 56)], bass="roots8", pedal="long",
         gtr=True, drums="big", timp=True)
    A.marker(T, "Bach's final cadence (m114-115)")
    A.mix(T, drums=86, lead=100, organ=104)
    cadence(A, FU, T)

    mf = A.save(out_path, "BWV 542 x U.N. Owen Was Her? (fused melody)")
    return A, mf


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    arr, mf = build(*sys.argv[1:])
    print(f"wrote {sys.argv[3]}: {len(arr.notes)} notes, {mf.length:.1f} s")
