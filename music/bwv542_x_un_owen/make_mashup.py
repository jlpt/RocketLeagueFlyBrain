#!/usr/bin/env python3
"""BWV 542 x U.N. Owen Was Her? -- a quodlibet: both melodies at the same time.

The fugue subject of J. S. Bach's Fantasia and Fugue in G minor (BWV 542)
and the main theme of ZUN's "U.N. Owen Was Her?" (Touhou 6) are played
*simultaneously*, over a new chord progression that fits both of them.

How the two tunes are locked together
-------------------------------------
* Owen is transposed from D minor to G minor (Bach's key) and kept at its
  original 155 bpm.  The fugue runs at half time (q = 77.5), so one Bach
  sixteenth = one Owen eighth and Owen's 8-bar main theme (A + A') lasts
  exactly as long as the fugue subject.
* Owen enters half a Bach beat after the subject's up-beat D.  Under the
  pair runs a new progression, one chord per Bach beat (two Owen beats):

    Gm  Gm | D7(#5)  C/D | Cm  F | Bb  Eb | D7sus4  D | Gm  C7 | C9  F | Gm
    (Bach's own circle-of-fifths sequence, and Owen's Dorian IV-chord)

  so every note of both tunes is a chord tone or a short passing tone.
  Owen's second (thirds) voice is kept only where it is a chord tone.
* The same lock is reused transposed (D minor, Owen's original key) for
  the fugue's *answer*, with the subject in the bass, and with the two
  tunes swapping instruments.
* Intro: Owen's 5/4 riff with the subject's head (D | Bb C A Bb G G') laid
  into each pair of riff cycles.  Bridge: Owen's chromatic theme against
  the subject's head sequenced over Owen's planing chords.  Ending: Bach's
  own final cadence (m114-115) with Owen's arpeggio over the G-major chord.

Sources (passed on the command line, not bundled):
  * BWV 542 fugue MIDI from the Mutopia Project edition (public domain):
    https://www.mutopiaproject.org/ftp/BachJS/BWV542/bwv542/bwv542-mids.zip
    -> bwv542-a4-1.mid
  * ZUN's SC-88Pro MIDI of U.N. Owen Was Her? (reference for melody and
    riff only; everything is re-harmonised and re-orchestrated).

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


def PC(names):
    """'G Bb D' -> {7, 10, 2}"""
    return {P(n + "4") % 12 for n in names.split()}


# ---------------------------------------------------------------- material
# BWV 542 fugue subject, soprano m1-4 (written pitch, times in Bach beats).
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
# after the subject: chord tones only (Owen holds his last note over Gm here)
CODETTA = {"minor": [(16, .5, "Bb5"), (16.5, .5, "G5"), (17, .5, "D5")],
           "major": [(16, .5, "B5"), (16.5, .5, "G5"), (17, .5, "D5")],
           "bass": [(16, 1.5, "G4")]}

# New harmony under both tunes (G minor), one chord per Bach beat:
#   (Bach beat, length, bass, pad voicing, pitch classes allowed for Owen's 2nd voice)
HARMONY = [
    (2, 2, "G2", ["G3", "Bb3", "D4"], "G Bb D A C"),        # Gm
    (4, 1, "D2", ["D3", "F#3", "C4"], "D F# Bb C E A"),     # D7(#5)
    (5, 1, "D2", ["C4", "E4", "G4"], "D C E G A"),          # C/D
    (6, 1, "C2", ["Eb3", "G3", "C4"], "C Eb G Bb D F A"),   # Cm(add9)
    (7, 1, "F2", ["F3", "C4"], "F A C G"),                  # F
    (8, 1, "Bb1", ["Bb3", "D4", "F4"], "Bb D F A C"),       # Bb
    (9, 1, "Eb2", ["Eb3", "G3", "Bb3"], "Eb G Bb D"),       # Eb
    (10, 1, "D2", ["D3", "G3", "C4"], "D G A C"),           # D7sus4
    (11, 1, "D2", ["F#3", "A3", "D4"], "D F# A"),           # D
    (12, 1, "G2", ["G3", "Bb3", "D4"], "G Bb D A"),         # Gm
    (13, 1, "C2", ["E3", "Bb3", "C4"], "C E G Bb D A"),     # C7
    (14, 1, "C2", ["E3", "Bb3", "D4"], "C E G Bb D A"),     # C9
    (15, 1, "F2", ["F3", "A3", "C4"], "F A C G"),           # F
    (16, 2, "G2", ["G3", "Bb3", "D4"], "G Bb D A"),         # Gm
]
FINAL_MAJOR = (16, 2, "G2", ["G3", "B3", "D4"], "G B D A")  # used before a key change


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


def owen_extract(notes, ch, a, b, transpose):
    """ZUN notes on `ch` in 16th-grid window [a, b) -> {rel16: [(dur16, pitch), ...]}"""
    out = {}
    for n in notes:
        if n["ch"] != ch:
            continue
        q = round(n["start"] * 4)
        if a <= q < b:
            out.setdefault(q - a, []).append((max(1, round(n["dur"] * 4)), n["pitch"] + transpose))
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- writing
CHANNELS = {
    # name: (channel, GM program, track name, volume, pan, reverb, chorus)
    "organ": (0, 19, "Church organ - manuals", 96, 60, 92, 0),
    "pedal": (1, 19, "Church organ - pedal", 100, 64, 88, 0),
    "lead":  (2, 80, "Square lead", 92, 68, 55, 30),
    "dbar":  (3, 16, "Drawbar organ", 70, 72, 55, 20),
    "hpsi":  (4, 6, "Harpsichord", 84, 36, 60, 0),
    "str":   (5, 48, "Strings", 74, 92, 85, 40),
    "bass":  (6, 33, "Finger bass", 96, 64, 25, 0),
    "cele":  (7, 8, "Celesta", 80, 96, 75, 0),
    "choir": (8, 52, "Choir aahs", 76, 64, 105, 30),
    "drums": (9, 16, "Drums (power kit)", 92, 64, 45, 0),
    "timp":  (10, 47, "Timpani", 96, 64, 80, 0),
    "gtr":   (11, 29, "Overdrive guitar", 70, 30, 45, 0),
    "brass": (12, 61, "Brass section", 72, 80, 70, 10),
    "glock": (13, 9, "Glockenspiel", 78, 100, 70, 0),
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
def riff(A, OW, T0, cycles, full):
    """Owen's 5/4 intro riff (Gm arpeggios + stabs) with the fugue subject's
    head (D | Bb C A Bb G G') laid into every pair of cycles on the organ."""
    CYC = 20
    arps, bassl = {}, {}
    src_a = owen_extract(OW, 2, 24, 24 + CYC * 4, -7)
    src_b = owen_extract(OW, 1, 24, 24 + CYC * 4, -7)
    for rep in range(0, cycles, 4):
        for src, dst in ((src_a, arps), (src_b, bassl)):
            for q, items in src.items():
                if q + rep * CYC < cycles * CYC:
                    dst[q + rep * CYC] = items
    for q, items in arps.items():
        k, pos = divmod(q, CYC)
        t = T0 + q * S16
        stab = len(items) > 1
        ps = [p for d, p in items]
        if stab and k == cycles - 1 and pos == 17:
            ps = [P("C4"), P("F#4"), P("A4")]           # last stab -> D7, sets up the subject's D
        dur = items[0][0] * S16 + (30 if stab else 0)
        A.chord("hpsi", t, dur, ps, 100 if stab else 80, human=4)
        if not stab and (full or k >= 2):
            A.note("cele", t, dur, ps[0] + 12, 60 + (8 if full else 3 * k))
        if stab and (full or k >= 4):
            A.chord("organ", t, items[0][0] * S16 + 30, ps, 86)
            A.chord("str", t, items[0][0] * S16 + 30, ps, 74)
            if full or k >= 6:
                A.chord("brass", t, items[0][0] * S16 + 30, [p + 12 for p in ps], 82)
    for q, items in bassl.items():
        k, pos = divmod(q, CYC)
        d, p = items[0]
        if k == cycles - 1 and pos == 16:
            p = P("D3")
        if full or k >= 2:
            A.note("bass", T0 + q * S16, d * S16 - 30, p, 94)
        if full or k >= 4:
            A.note("pedal", T0 + q * S16, d * S16 - 30, p - 12, 74)
    if not full:
        A.note("pedal", T0, 4 * CYC * S16 - 40, P("G1"), 76)
        A.note("pedal", T0, 4 * CYC * S16 - 40, P("G2"), 58)
    # subject head: up-beat D on the last beat of an even cycle, the rest in the next cycle
    head = [(0, 2, "Bb3"), (2, 2, "C4"), (4, 2, "A3"), (6, 2, "Bb3"), (8, 4, "G3"), (12, 4, "G4")]
    for k in range(0, cycles, 2):
        t = T0 + k * CYC * S16
        v = 92 if full or k >= 2 else 84
        A.note("organ", t + 16 * S16, 4 * S16 - 20, P("D4"), v)
        for q, d, nm in head:
            A.note("organ", t + (CYC + q) * S16, d * S16 - 20, P(nm), v)
    for k in range(cycles):
        t = T0 + k * CYC * S16
        if k == cycles - 1:
            A.drums(t, {K: "X.....X.....X.X.....", CH: "x.x.x.x.x.x.........",
                        SN: "....x.....x.oxxxXXXX", TMID: "............x......."})
        elif full or k >= 4:
            A.drums(t, {K: "X.....X.....X.X..X..", SN: "....x.....x......X..", CH: "x.x.x.x.x.x.........",
                        OH: "............x.......", CR: "X" if k % 2 == 0 else ""})
        elif k >= 2:
            A.drums(t, {RD: "x...x...x...x.......", K: "X...........x.x..x..", PH: "..o...o...o........."}, 0.85)


def quodlibet(A, U, tr, opt):
    """Bach's subject and Owen's main theme together, 8 bars from tick U
    (= Owen's first note = Bach beat 2).  `tr` transposes everything."""
    def bt(b):
        return U + (b - 2) * BB

    harmony = HARMONY[:-1] + [FINAL_MAJOR if opt.get("final_major") else HARMONY[-1]]
    tones_at = {}
    for b, ln, bass, pad, tones in harmony:
        for i in range(ln * 8):
            tones_at[(b - 2) * 8 + i] = {(pc + tr) % 12 for pc in PC(tones)}

    # --- Bach: subject (+ codetta)
    subj = SUBJECT + CODETTA[opt.get("codetta") or ("major" if opt.get("final_major") else "minor")]
    bach, owen = [], []                               # sounding melody notes: (start, end, pitch)
    for i, (b, d, nm) in enumerate(subj):
        p = P(nm) + tr
        if i == 0 and opt.get("tonal_answer"):
            p -= 2                                    # Bach's tonal answer starts on G, not A
        for inst, octv, vel in opt["subject"]:
            A.note(inst, bt(b), d * BB - 24, p + octv, vel, human=2)
            bach.append((bt(b), bt(b) + d * BB, p + octv))

    def clashes(t0, t1, pitch, against):
        return any(s < t1 and e > t0 and harsh(pitch, m) for s, e, m in against)
    # --- Owen: main theme (A + A'), 2nd voice only where it is a chord tone
    for u, items in opt["owen_notes"].items():
        ps = sorted(p + tr for d, p in items)
        d = items[0][0]
        t = U + u * S16
        dur = d * S16 - (0 if d >= 8 else 20)
        for inst, octv, vtop, vlow in opt["owen"]:
            A.note(inst, t, dur, ps[-1] + octv, vtop)
            owen.append((t, t + dur, ps[-1] + octv))
            if (vlow and len(ps) > 1 and ps[0] % 12 in tones_at[u]
                    and not clashes(t, t + dur, ps[0] + octv, bach)):
                A.note(inst, t, dur, ps[0] + octv, vlow)
                owen.append((t, t + dur, ps[0] + octv))
    # --- harmony
    for b, ln, bass, pad, tones in harmony:
        t, dur = bt(b), ln * BB
        bp = P(bass) + tr
        while bp > P("D2"):
            bp -= 12
        while bp < P("Bb1") - 1:
            bp += 12
        padp = [P(x) + tr for x in pad]
        while min(padp) < P("D3"):                   # keep pads in the D3..C#4 window
            padp = [x + 12 for x in padp]
        while min(padp) > P("C#4"):
            padp = [x - 12 for x in padp]
        for inst, octv, vel in opt.get("pad", []):
            # drop pad notes that would rub (m2/m9) against a passing note of either tune
            keep = [x + octv for x in padp if not clashes(t, t + dur, x + octv, bach + owen)]
            A.chord(inst, t, dur - 15, keep or [min(padp) + octv], vel)
        if opt.get("bass") == "roots":
            for e in range(ln * 4):
                A.note("bass", t + e * 2 * S16, 2 * S16 - 30, bp + (12 if e % 2 else 0), 96 if e % 2 == 0 else 84)
        elif opt.get("bass") == "long":
            A.note("bass", t, dur - 30, bp, 84)
        if opt.get("pedal_roots"):
            A.note("pedal", t, dur - 20, bp, 80)
            A.note("pedal", t, dur - 20, bp + 12, 62)
        if opt.get("timp") and (b - 2) % 4 == 0:
            A.note("timp", t, Q, bp if bp >= P("F2") else bp + 12, 92)
    # --- drums: 8 bars from U
    style = opt.get("drums")
    for bar in range(8):
        t = U + bar * 16 * S16
        if style == "light":
            pat = {CH: "o.x.o.x.o.x.o.x.", K: "x.......x......."}
            if bar == 0:
                pat[CR] = "x"
        elif style in ("groove", "big"):
            pat = dict(MAIN_GROOVE)
            if style == "big":
                pat[CH] = "x.o.x.o.x.o.x.o."
            if bar in ((0, 4) if style == "groove" else (0, 2, 4, 6)):
                pat[CR] = "X"
        else:
            continue
        if bar == 7:
            pat[SN] = "........x.xxXXXX" if style == "big" else "..........x.xxXx"
            pat.pop(CP, None)
        A.drums(t, pat)


def bridge(A, OW, T0):
    """Owen's chromatic theme against the subject's head, sequenced over
    Owen's planing chords (Gm F#m A Ab); ends on D7 for the subject's up-beat."""
    cyc = [("G", ["G3", "D4", "G4"], ["Bb3", "C4", "A3", "Bb3", "G3", "G4"]),
           ("F#", ["F#3", "C#4", "F#4"], ["A3", "B3", "G#3", "A3", "F#3", "F#4"]),
           ("A", ["A3", "E4", "A4"], ["C#4", "D4", "B3", "C#4", "A3", "A4"]),
           ("Ab", ["Ab3", "Eb4", "Ab4"], ["C4", "Db4", "Bb3", "C4", "Ab3", "Ab4"])]
    d7 = ("D", ["D3", "A3", "C4"], ["F#3", "G3", "E3", "F#3"])
    rhythm = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 2), (6, 2)]
    for i in range(16):
        root, pad, head = d7 if i == 15 else cyc[i % 4]
        t = T0 + i * 8 * S16
        A.chord("str", t, 8 * S16 - 15, [P(x) for x in pad], 70 if i < 8 else 78)
        A.chord("organ", t, 8 * S16 - 15, [P(x) for x in pad], 58)
        for (q, d), nm in zip(rhythm, head):
            A.note("organ", t + q * S16, d * S16 - 15, P(nm), 94)
            if i >= 8:
                A.note("hpsi", t + q * S16, d * S16 - 15, P(nm), 80)
        r = P(root + "2")
        if r > P("A2"):
            r -= 12
        for e in range(4):
            A.note("bass", t + e * 2 * S16, 2 * S16 - 30, r + (12 if e % 2 else 0), 94 if e % 2 == 0 else 82)
    mel = owen_extract(OW, 6, 248, 312, +5)
    for rep in range(2):
        for u, items in mel.items():
            ps = sorted(p for d, p in items)
            d = items[0][0]
            t = T0 + (rep * 64 + u) * S16
            A.note("lead", t, d * S16, ps[0], 100)
            A.note("glock", t, d * S16, ps[0], 74)
            if rep:
                A.note("dbar", t, d * S16, ps[0], 78)
                A.note("cele", t, d * S16 + S16, ps[-1], 72)
    for bar in range(8):
        pat = {K: "X...X...X...X...", RS: "x...x...x...x...", CH: "X.X...XXX...X..X", OH: "....x.....x...x."}
        if bar >= 4:
            pat[CP] = "....x.......x..."
        if bar in (0, 4):
            pat[CR] = "X"
        if bar == 7:
            pat = {K: "X...X...X...X...", CH: "X.X...XXX.......", SN: "....x...x.xxXXXX",
                   TLO: "..........x.x...", TMID: "..........x.x..."}
        A.drums(T0 + bar * 16 * S16, pat, 0.95)


def cadence(A, FU, T0):
    """Bach's own final bar (m114) and G-major chord, with Owen's arpeggio on top."""
    def ct(beat):
        return T0 + (beat - 452) * BB
    hold = 4 * BB
    for n in FU:
        if n["track"] not in (1, 3, 5) or not 452 <= n["start"] < 457:
            continue
        s, d, p = n["start"], n["dur"], n["pitch"]
        final = s >= 456
        dd = (hold if final else d * BB) - 24
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
    for i, nm in enumerate(["B5", "G5", "D5"] * 4):        # Owen's riff, now in G major
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

    # sanity check: the hard-coded subject matches the edition (top voice of the manuals)
    top = {}
    for n in FU:
        if n["track"] == 1 and n["start"] < 16:
            top[n["start"]] = max(top.get(n["start"], 0), n["pitch"])
    for b, d, nm in SUBJECT:
        assert top.get(b) == P(nm), (b, nm, top.get(b))

    owen_theme = owen_extract(OW, 6, 312, 440, -7)       # main theme A + A', G minor
    T = 0
    A.tempo(0, 155)

    # 1. intro: Owen's 5/4 riff + subject head
    A.marker(T, "Intro: Owen 5/4 riff + Bach subject head")
    A.timesig(T, 5)
    A.mix(T, drums=84)
    riff(A, OW, T, 8, full=False)
    T += 8 * 20 * S16
    A.timesig(T, 4)

    def unit(label, tr, **opt):
        nonlocal T
        A.marker(T, label)
        opt["owen_notes"] = owen_theme
        quodlibet(A, T, tr, opt)
        T += 32 * Q

    # 2. Q1: subject on organ, Owen on glockenspiel/celesta, light
    A.mix(T, drums=76, organ=92, glock=100, cele=96)
    unit("Q1 (G minor): subject on organ + Owen on bells", 0,
         subject=[("organ", 0, 94), ("hpsi", 0, 70)],
         owen=[("glock", 0, 96, 70), ("cele", 0, 84, 0)],
         pad=[("str", 0, 58)], bass="long", drums="light")
    # 3. Q2: the fugue's answer in D minor + Owen in its original key
    A.mix(T, drums=84, lead=86, organ=104)
    unit("Q2 (D minor): answer on organ + Owen in original key", -5,
         subject=[("organ", 0, 96)],
         owen=[("lead", 0, 100, 82), ("dbar", 0, 80, 66)],
         pad=[("str", 0, 66)], bass="roots", drums="groove", tonal_answer=True, final_major=True)
    # 4. Q3: subject in the bass
    A.mix(T, drums=86, lead=96, organ=84)
    unit("Q3 (G minor): subject in the bass + Owen on lead", 0,
         subject=[("pedal", -24, 96), ("pedal", -12, 70), ("bass", -24, 98), ("gtr", -12, 64)], codetta="bass",
         owen=[("lead", 0, 102, 84), ("dbar", 0, 82, 68)],
         pad=[("organ", 0, 74), ("str", 12, 62)], drums="groove", timp=True)
    # 5. riff reprise
    A.marker(T, "Riff reprise")
    A.timesig(T, 5)
    A.mix(T, drums=86, organ=100)
    riff(A, OW, T, 4, full=True)
    T += 4 * 20 * S16
    A.timesig(T, 4)
    # 6. bridge
    A.marker(T, "Bridge: Owen chromatic theme + subject head")
    A.mix(T, drums=86, lead=96, organ=96)
    bridge(A, OW, T)
    T += 32 * Q
    # 7. Q4: swap - subject on the synth lead, Owen on the organ
    A.mix(T, drums=86, lead=92, organ=100, cele=88)
    unit("Q4 (G minor): swapped - subject on synth, Owen on organ", 0,
         subject=[("lead", 0, 96), ("dbar", 0, 78)],
         owen=[("organ", 0, 92, 78), ("cele", 0, 70, 0)],
         pad=[("str", 0, 66)], bass="roots", drums="groove")
    # 8. Q5: climax
    A.mix(T, drums=90, lead=94, organ=112, glock=84)
    A.ramp("choir", T, T + 16 * Q, 11, 70, 120)
    unit("Q5 (G minor): climax - subject on organ + guitar, Owen on lead", 0,
         subject=[("organ", 0, 100), ("gtr", 0, 74)],
         owen=[("lead", 0, 106, 88), ("dbar", 0, 84, 70), ("glock", 0, 70, 0)],
         pad=[("choir", 0, 74), ("str", 12, 66), ("brass", 0, 56)], bass="roots",
         pedal_roots=True, drums="big", timp=True)
    # 9. Q6: finale, subject in the bass again under everything
    A.mix(T, drums=90, lead=98, organ=80)
    unit("Q6 (G minor): finale - subject in the bass, Owen on top", 0,
         subject=[("pedal", -24, 100), ("pedal", -12, 74), ("bass", -24, 100), ("gtr", -12, 76)], codetta="bass",
         owen=[("lead", 0, 108, 90), ("dbar", 0, 86, 72), ("glock", 0, 72, 0)],
         pad=[("organ", 0, 84), ("choir", 0, 76), ("str", 12, 70)], drums="big", timp=True)
    # 10. Bach's final cadence
    A.marker(T, "Bach's final cadence (m114-115)")
    A.mix(T, drums=92, lead=100, organ=104)
    end = cadence(A, FU, T)

    mf = A.save(out_path, "BWV 542 x U.N. Owen Was Her? (quodlibet)")
    return A, mf, end


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    arr, mf, end = build(*sys.argv[1:])
    print(f"wrote {sys.argv[3]}: {len(arr.notes)} notes, {mf.length:.1f} s")
