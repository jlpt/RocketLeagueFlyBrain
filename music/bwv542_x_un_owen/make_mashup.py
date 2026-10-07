#!/usr/bin/env python3
"""BWV 542 x U.N. Owen Was Her? -- a MIDI mashup generator.

Builds a ~3.5 minute arrangement in G minor that splices and layers
J. S. Bach's Fantasia and Fugue in G minor (BWV 542) with ZUN's
"U.N. Owen Was Her?" (Touhou 6, Flandre Scarlet's theme).

Sources (passed on the command line, not bundled):
  * BWV 542 Fantasia / Fugue MIDIs from the Mutopia Project edition
    (public domain, typeset by Urs Metzger):
    https://www.mutopiaproject.org/ftp/BachJS/BWV542/bwv542/bwv542-mids.zip
    -> bwv542.mid (Fantasia) and bwv542-a4-1.mid (Fugue)
  * ZUN's SC-88Pro MIDI of U.N. Owen Was Her? (used only as a reference
    for melody, bass line and harmony; everything is re-orchestrated).

Usage:
  python3 make_mashup.py FANTASIA.mid FUGUE.mid UN_OWEN_ZUN.mid OUT.mid

Form (all in G minor; Owen is transposed down a fifth from D minor):
  A  Fantasia bars 1-3, organ + choir, rubato          -> F#dim7 over G pedal
  B  Owen intro riff (5/4), harpsichord/celesta build   -> Gm
  C  Owen "chromatic" theme over planing organ chords   -> Ab (bII)
  D  Owen main theme x2; harpsichord counter-melody built
     from the fugue subject's turn figure (Bb C A Bb G G')
  E  Fugue exposition (m1-22) at half time (q=77.5) with the band
     entering voice by voice; ends on a D7 half cadence
  F  Owen main theme climax (deceptive V-VI into Eb): organ plays the
     Bach counter-melody, choir, guitar, full kit
  G  Fugue ending (m106-115) incl. the final pedal entry, ritardando,
     Bach's G-major final chord
"""
import random
import sys

import mido

TPB = 480
Q = TPB
S16 = TPB // 4

_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def P(name):
    """'Bb4' -> 70 (MIDI note number, C4 = 60)."""
    i, acc = 1, 0
    while name[i] in "#b":
        acc += 1 if name[i] == "#" else -1
        i += 1
    return 12 * (int(name[i:]) + 1) + _STEP[name[0]] + acc


# ---------------------------------------------------------------- reading
def read_notes(path):
    """All notes of a MIDI file as dicts (start/dur in quarter notes)."""
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


# ---------------------------------------------------------------- writing
CHANNELS = {
    # name: (channel, GM program, track name, volume, pan, reverb, chorus)
    "organ": (0, 19, "Church organ - manuals", 96, 64, 96, 0),
    "pedal": (1, 19, "Church organ - pedal", 100, 64, 90, 0),
    "lead":  (2, 80, "Square lead", 82, 58, 55, 30),
    "dbar":  (3, 16, "Drawbar organ (lead double)", 66, 72, 55, 20),
    "hpsi":  (4, 6, "Harpsichord", 84, 36, 60, 0),
    "str":   (5, 48, "Strings", 78, 92, 85, 40),
    "bass":  (6, 33, "Finger bass", 96, 64, 25, 0),
    "cele":  (7, 8, "Celesta", 72, 96, 75, 0),
    "choir": (8, 52, "Choir aahs", 80, 64, 105, 30),
    "drums": (9, 16, "Drums (power kit)", 104, 64, 45, 0),
    "timp":  (10, 47, "Timpani", 98, 64, 80, 0),
    "gtr":   (11, 29, "Overdrive guitar", 62, 30, 45, 0),
    "brass": (12, 61, "Brass section", 74, 80, 70, 10),
}

K, K2, RS, SN, CP, CH, PH, OH = 36, 35, 37, 38, 39, 42, 44, 46
CR, CR2, RD, SPL, CHN = 49, 57, 51, 55, 52
TLO, TMID, THI, TFL = 45, 47, 50, 41


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

    def chord(self, name, tick, dur, pitches, vel, human=4):
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

    def drums(self, t0, patterns, vel_scale=1.0, step=S16):
        """patterns: {drum note: 'X.x.o...'}; X=accent, x=normal, o=ghost."""
        vmap = {"X": 112, "x": 88, "o": 58}
        for drum, pat in patterns.items():
            for i, c in enumerate(pat):
                if c in vmap:
                    self.note("drums", t0 + i * step, S16, drum, vmap[c] * vel_scale, human=5)

    def timp_roll(self, t0, t1, pitch, v0, v1, rate=S16 // 2):
        n = max(1, int((t1 - t0) // rate))
        for i in range(n):
            self.note("timp", t0 + i * rate, rate, pitch, v0 + (v1 - v0) * i / max(1, n - 1), human=3)

    def save(self, path, title):
        # resolve overlapping notes of the same pitch on the same channel
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


# ------------------------------------------------------------ material
def owen_extract(notes, ch, a, b, transpose):
    """ZUN notes on `ch` in 16th-grid window [a, b): (rel16, dur16, pitch)."""
    out = []
    for n in notes:
        if n["ch"] != ch:
            continue
        q = round(n["start"] * 4)
        if a <= q < b:
            out.append((q - a, max(1, round(n["dur"] * 4)), n["pitch"] + transpose))
    return sorted(out)


def group_onsets(evts):
    g = {}
    for q, d, p in evts:
        g.setdefault(q, []).append((d, p))
    return g


# Owen main-theme harmony in G minor: one chord per half bar (8 sixteenths)
#            name, root pc-name, pad voicing, bass pair (root, fifth)
EB = ("Eb", ["Eb3", "Bb3", "G4"], ["Eb2", "Bb2"])
F_ = ("F", ["F3", "C4", "A4"], ["F2", "C3"])
GM = ("Gm", ["G3", "D4", "Bb4"], ["G2", "D3"])
CE = ("C/E", ["E3", "C4", "C5"], ["E2", "C3"])
EB2 = ("Eb", ["Eb3", "Bb3", "Eb5"], ["Eb2", "Bb2"])
F2 = ("F", ["F3", "C4", "F5"], ["F2", "C3"])
GM2 = ("Gm", ["G3", "D4", "G5"], ["G2", "D3"])
PHRASE_CHORDS = [EB, F_, GM, CE, EB2, F2, GM2, GM2]
CHORD_ROOT = {"Eb": "Eb", "F": "F", "Gm": "G", "C/E": "C"}
GTR_ROOT = {"Eb": "Eb3", "F": "F3", "Gm": "G3", "C/E": "C3"}
CHOIR_VOICING = {"Eb": ["Eb3", "G3", "Bb3", "Eb4", "G4"], "F": ["F3", "A3", "C4", "F4", "A4"],
                 "Gm": ["G3", "Bb3", "D4", "G4", "Bb4"], "C/E": ["E3", "G3", "C4", "E4", "G4"]}

# Counter-melody made of the BWV 542 fugue subject's cells, re-fitted to
# Owen's chords.  (rel16, dur16, note).  The Gm cell is Bach's literal
# opening "Bb C A Bb G G'"; the C/E cell mirrors "F# G E F# D G D".
_C_COMMON = [
    (0, 1, "G4"), (1, 1, "A4"), (2, 1, "F4"), (3, 1, "G4"), (4, 2, "Eb4"), (6, 2, "Eb5"),      # Eb
    (8, 1, "A4"), (9, 1, "Bb4"), (10, 1, "G4"), (11, 1, "A4"), (12, 2, "F4"), (14, 2, "F5"),   # F
    (16, 1, "Bb4"), (17, 1, "C5"), (18, 1, "A4"), (19, 1, "Bb4"), (20, 2, "G4"), (22, 2, "G5"),  # Gm
    (24, 1, "E5"), (25, 1, "F5"), (26, 1, "D5"), (27, 1, "E5"), (28, 2, "C5"), (30, 1, "G5"), (31, 1, "C5"),  # C/E
    (32, 1, "G5"), (33, 1, "A5"), (34, 1, "F5"), (35, 1, "G5"), (36, 2, "Eb5"), (38, 2, "Eb4"),  # Eb
    (40, 1, "C5"), (41, 1, "D5"), (42, 1, "Bb4"), (43, 1, "C5"), (44, 2, "F4"), (46, 2, "F5"),   # F
]
COUNTER_A = _C_COMMON + [
    (48, 1, "D5"), (49, 1, "C5"), (50, 1, "Bb4"), (51, 1, "A4"), (52, 2, "G4"), (54, 2, "Bb4"),  # Gm
    (56, 2, "D5"), (58, 2, "Bb4"), (60, 2, "G4"), (62, 1, "D4"), (63, 1, "F#4"),                # Gm
]
COUNTER_B = _C_COMMON + [
    (48, 1, "Bb4"), (49, 1, "C5"), (50, 1, "A4"), (51, 1, "Bb4"), (52, 2, "G4"), (54, 2, "G5"),  # Gm
    (56, 1, "Bb4"), (57, 1, "A4"), (58, 1, "G4"), (59, 1, "F#4"), (60, 2, "G4"), (62, 2, "D4"),  # Gm
]

MAIN_GROOVE = {K: "X...X...X...X...", CH: "x...x...x...x...", OH: "..x...x...x...x.",
               CP: "....x.......x..."}


def build(fantasia_path, fugue_path, owen_path, out_path):
    A = Arrangement()
    FA = read_notes(fantasia_path)
    FU = read_notes(fugue_path)
    OW = read_notes(owen_path)

    # ============================================================ A: Fantasia
    A.marker(0, "Fantasia (BWV 542, bars 1-3)")
    A.timesig(0, 4)
    A.tempo(0, 46)
    A.tempo(8 * Q, 43)
    A.tempo(10 * Q, 40)
    A.tempo(11 * Q, 35)
    A.tempo(11.5 * Q, 30)
    end_a = 12 * Q
    for n in FA:
        if n["start"] >= 12 or n["track"] not in (1, 2, 5):
            continue
        t, d, p = n["start"] * Q, n["dur"] * Q, n["pitch"]
        if n["track"] == 2 and n["start"] >= 10:
            d = end_a - t                      # let the F#dim7 ring into the riff
        if n["track"] == 5:
            d = end_a - t if n["start"] >= 10 else d
            A.note("pedal", t, d - 10, p, 92)
            A.note("pedal", t, d - 10, p + 12, 70)
        else:
            A.note("organ", t, d - 10, p, 90, human=2)
        if n["track"] == 2:                    # choir doubles the left-hand chords as a pad
            A.note("choir", t, max(d, 2 * Q) - 20, p, 72, human=2)
    A.ramp("choir", 0, 4 * Q, 11, 70, 115)
    A.ramp("choir", 8 * Q, 12 * Q, 11, 115, 80)
    A.timp_roll(0, 1.5 * Q, P("G2"), 96, 50, rate=Q // 8)
    A.timp_roll(11 * Q, 12 * Q, P("G2"), 45, 100, rate=Q // 8)

    # ============================================================ B: Owen intro (5/4)
    B0 = end_a
    A.marker(B0, "U.N. Owen intro riff (5/4)")
    A.mix(B0, drums=86)
    A.tempo(B0, 155)
    A.timesig(B0, 5)
    CYC = 20
    riff = owen_extract(OW, 2, 24, 184, -7)          # arps + stabs, now Bb4 G4 D4 / C F A
    bassl = owen_extract(OW, 1, 24, 184, -7)
    groups = group_onsets(riff)
    A.note("pedal", B0, 2 * CYC * S16 - 40, P("G1"), 80)
    A.note("pedal", B0, 2 * CYC * S16 - 40, P("G2"), 60)
    A.note("timp", B0, Q, P("G2"), 105)
    A.drums(B0, {CR: "X"})
    for q, items in groups.items():
        k = q // CYC
        t = B0 + q * S16
        stab = len(items) > 1
        for d, p in items:
            A.note("hpsi", t, (d + 1) * S16, p, 100 if stab else 80)
            if not stab and k >= 2:
                A.note("cele", t, (d + 1) * S16, p + 12, 62 + 3 * k)
        if stab and k >= 4:
            ps = [p for d, p in items]
            dur = items[0][0] * S16 + 30
            A.chord("organ", t, dur, ps, 96)
            if k >= 6:
                A.chord("brass", t, dur, [p + 12 for p in ps], 88)
            A.chord("str", t, dur, ps, 80)
    for q, d, p in bassl:
        k = q // CYC
        t = B0 + q * S16
        if k >= 2:
            A.note("bass", t, d * S16 - 30, p, 96)
        if k >= 4:
            A.note("pedal", t, d * S16 - 30, p - 12, 78)
            if q % CYC in (0, 12):
                A.note("timp", t, Q, p if p < P("D3") else p - 12, 92)
    for k in range(8):
        t = B0 + k * CYC * S16
        if k in (2, 3):
            A.drums(t, {RD: "x...x...x...x.......", K: "X...........x.x..x..", PH: "..o...o...o........."}, 0.85)
        elif 4 <= k <= 6:
            A.drums(t, {K: "X.....X.....X.X..X..", SN: "....x.....x......X..", CH: "x.x.x.x.x.x.........",
                        OH: "............x......."[:CYC], CR: ("X..........." + "X.......") if k in (4, 6) else ""})
        elif k == 7:
            A.drums(t, {K: "X.....X.....X.X..X..", CH: "x.x.x.x.x.x.........",
                        SN: "....x.....x.oxxxXXXX", TMID: "............x......."})

    # ============================================================ C: chromatic theme
    C0 = B0 + 8 * CYC * S16
    A.marker(C0, "U.N. Owen chromatic theme")
    A.mix(C0, drums=94, lead=106, dbar=74)
    A.timesig(C0, 4)
    chords = owen_extract(OW, 3, 184, 312, +5)       # G3 D4 Bb4 / F#3 C#4 A4 / A3 E4 C#5 / Ab3 Eb4 C5
    cg = group_onsets(chords)
    for q, items in sorted(cg.items()):
        t = C0 + q * S16
        ps = sorted(p for d, p in items)
        dur = items[0][0] * S16
        A.chord("organ", t, dur - 12, ps, 86, human=2)
        A.note("pedal", t, dur - 12, ps[0] - 12, 82)
        A.note("pedal", t, dur - 12, ps[0] - 24, 64)
        arp = [ps[2], ps[0] + 12, ps[1]]
        if arp[0] < arp[1]:
            arp[0] += 12
        for i in range(8):
            A.note("hpsi", t + i * S16, S16 * (2 if i < 7 else 1) - 10, arp[i % 3], (78 if q < 64 else 62) - (8 if i % 3 else 0))
        if q >= 64:
            A.chord("str", t, dur - 12, [p + 12 for p in ps], 66)
    for q, d, p in owen_extract(OW, 1, 184, 312, -7):
        A.note("bass", C0 + q * S16, d * S16 - 25, p, 94)
    mel = group_onsets(owen_extract(OW, 6, 248, 312, +5))
    for q, items in mel.items():
        ps = sorted(p for d, p in items)
        d = items[0][0]
        t = C0 + (q + 64) * S16
        A.note("lead", t, d * S16, ps[0], 98)
        A.note("dbar", t, d * S16, ps[0], 80)
        A.note("cele", t, d * S16 + S16, ps[-1], 80)
    for bar in range(8):
        t = C0 + bar * 16 * S16
        pat = {K: "X...X...X...X...", RS: "x...x...x...x...", CH: "X.X...XXX...X..X", OH: "....x.....x...x."}
        if bar >= 4:
            pat[CP] = "....x.......x..."
        if bar in (0, 4):
            pat[CR] = "X"
        if bar == 3:
            pat[SN] = "............o.xx"
        if bar == 7:
            pat = {K: "X...X...X...X.X.", CH: "X.X...XXX.......", SN: "....x.......xxxX",
                   TLO: "........x.x.....", TMID: "..........x.x..."}
        A.drums(t, pat, 0.9)
    A.note("timp", C0, Q, P("G2"), 100)

    # ======================================================= Owen main theme helper
    def main_theme(T0, climax):
        lead = group_onsets(owen_extract(OW, 6, 312, 568, -7))
        for q, items in lead.items():
            ps = sorted(p for d, p in items)
            d = items[0][0]
            t = T0 + q * S16
            long_note = d >= 8
            A.note("lead", t, d * S16 - (0 if long_note else 20), ps[-1], 104 if climax else 100)
            A.note("lead", t, d * S16 - (0 if long_note else 20), ps[0], 84 if climax else 80)
            A.note("dbar", t, d * S16 - (0 if long_note else 20), ps[-1], 84)
            A.note("dbar", t, d * S16 - (0 if long_note else 20), ps[0], 70)
            if climax:
                A.note("brass", t, d * S16 - 20, ps[-1] - 12, 70 if q < 128 else 82)
        for ph in range(4):
            P0 = T0 + ph * 64 * S16
            counter = COUNTER_A if ph % 2 == 0 else COUNTER_B
            for ci, (cname, voicing, bpair) in enumerate(PHRASE_CHORDS):
                if ci == 7:   # last Gm is tied to the previous one
                    continue
                t = P0 + ci * 8 * S16
                dur = (16 if ci == 6 else 8) * S16
                A.chord("str", t, dur - 15, [P(v) + (12 if climax else 0) for v in voicing],
                        70 if climax else 72, human=2)
                if climax:
                    A.chord("choir", t, dur - 15, [P(v) for v in CHOIR_VOICING[cname]], 74, human=2)
                    A.note("pedal", t, dur - 15, P(bpair[0]), 84)
                    A.note("pedal", t, dur - 15, P(bpair[0]) - 12, 66)
                final_bar = (ph == 3 and ci == 6 and not climax)
                if final_bar:   # one big G-minor hit with the crash, then let it ring
                    A.note("bass", t, 16 * S16, P("G1"), 100)
                    A.note("timp", t, Q, P("G2"), 108)
                for half in range(0 if final_bar else (1 if ci != 6 else 2)):
                    root = GTR_ROOT[cname]
                    for e in range(4):
                        tt = t + (half * 4 + e) * 2 * S16
                        A.note("bass", tt, 2 * S16 - 30, P(bpair[e % 2]), 98 if e % 2 == 0 else 86)
                        if climax:
                            A.chord("gtr", tt, int(1.5 * S16), [P(root), P(root) + 7, P(root) + 12],
                                    84 if e % 2 == 0 else 70)
                if ci in (0, 4) or (ci == 2 and climax):
                    A.note("timp", t, Q, P(bpair[0]) + (12 if P(bpair[0]) < P("D2") else 0), 96 if climax else 84)
            # Bach counter-melody: harpsichord in 2nd half of D; organ (+harpsichord 8va) in F
            if climax or ph >= 2:
                for q, d, nm in counter:
                    t = P0 + q * S16
                    if climax:
                        A.note("organ", t, d * S16 - 12, P(nm), 96, human=3)
                        if ph >= 2:
                            A.note("hpsi", t, d * S16 - 10, P(nm) + 12, 84)
                    else:
                        A.note("hpsi", t, d * S16 - 10, P(nm), 92)
            else:
                for ci, (cname, voicing, bpair) in enumerate(PHRASE_CHORDS):
                    t = P0 + ci * 8 * S16
                    tones = sorted(P(v) % 12 for v in CHOIR_VOICING[cname][:3])
                    trip = sorted(set((P("D4") + ((pc - P("D4")) % 12)) for pc in tones))
                    arp = [trip[2], trip[1], trip[0]]
                    for i in range(8):
                        A.note("hpsi", t + i * S16, (2 if i < 7 else 1) * S16 - 10, arp[i % 3], 74 - (6 if i % 3 else 0))
            # drums
            for bar in range(4):
                t = P0 + bar * 16 * S16
                pat = dict(MAIN_GROOVE)
                if climax:
                    pat[CH] = "x.o.x.o.x.o.x.o."
                if bar == 0:
                    pat[CR] = "X"
                    if climax:
                        pat[CR2] = "X"
                if climax and bar == 2:
                    pat[CR] = "X"
                if bar == 3:
                    pat[SN] = "..........x..x.x"
                    if ph == 3 and climax:
                        pat = {K: "X...X...X...X...", CH: "x...x...", SN: "....x...xxxxXXXX",
                               THI: "............x...", TMID: ".............x..", TLO: "..............x."}
                    elif ph == 3:   # stop on a hit and let the chord ring into the fugue
                        pat = {K: "X", CR: "X", CR2: "X"}
                A.drums(t, pat, 1.0 if climax else 0.92)
        if climax:
            A.ramp("choir", T0, T0 + 4 * 16 * S16, 11, 80, 120)

    # ============================================================ D: main theme
    D0 = C0 + 8 * 16 * S16
    A.marker(D0, "U.N. Owen main theme (Bach counter-melody on harpsichord)")
    A.mix(D0, drums=92, lead=104, dbar=70)
    main_theme(D0, climax=False)
    A.note("timp", D0, Q, P("Eb2") + 12, 100)

    # ============================================================ E: fugue exposition
    E0 = D0 + 16 * 16 * S16
    A.marker(E0, "Fugue exposition (BWV 542, m1-22)")
    A.mix(E0, drums=80, lead=90)
    BB = 2 * Q                      # one Bach beat = two MIDI beats (half time)
    CUT = 87.0

    def bt(beat, base):
        return base + beat * BB

    for n in FU:
        if n["track"] not in (1, 3, 5) or n["start"] >= CUT:
            continue
        s, d, p = n["start"], min(n["dur"], CUT - n["start"]), n["pitch"]
        t, dd = bt(s, E0), d * BB - 24
        if n["track"] == 5:
            if s >= 86:
                continue                                 # replaced by the held D below
            A.note("pedal", t, dd, p, 88)
            A.note("pedal", t, dd, p - 12, 70)
            A.note("bass", t, dd - 10, p, 92)            # bass doubles the pedal from its entry (m14)
            if s < 69:
                A.note("gtr", t, dd - 10, p + 12, 82)    # ... and overdrive guitar its subject
            if s >= 56 and (s % 2 == 0) and d >= 0.5:
                A.note("timp", t, Q, p if p < P("D3") else p - 12, 84)
        else:
            A.note("organ", t, dd, p, 90, human=2)
            if n["track"] == 3:
                if 35.5 <= s < 49:
                    A.note("lead", t, dd, p + 24, 72)    # square lead highlights the tenor subject
                if 35.5 <= s < 56:
                    A.note("bass", t, dd - 10, p - 12, 84)
            if s >= 36 and d >= 1.0:
                A.note("str", t, dd, p + (12 if n["track"] == 3 else 0), 68)
    # held D7 half cadence (beats 87-90) under the soprano's "F# G E F# D"
    hold0, hold1 = bt(CUT, E0), bt(90, E0)
    A.chord("organ", hold0, hold1 - hold0 - 20, [P("C4"), P("F#4"), P("A4"), P("D5")], 96)
    A.note("pedal", bt(86, E0), hold1 - bt(86, E0) - 20, P("D3"), 92)
    A.note("pedal", bt(86, E0), hold1 - bt(86, E0) - 20, P("D2"), 76)
    A.note("bass", bt(86, E0), hold1 - bt(86, E0) - 20, P("D2"), 96)
    A.chord("choir", hold0, hold1 - hold0 - 20, [P("D3"), P("A3"), P("C4"), P("F#4"), P("A4"), P("D5")], 80)
    A.chord("str", hold0, hold1 - hold0 - 20, [P("D4"), P("F#4"), P("C5"), P("D5"), P("F#5")], 76)
    A.ramp("choir", hold0, hold1, 11, 50, 127)
    A.ramp("str", hold0, hold1, 11, 70, 127)
    A.timp_roll(bt(87.5, E0), hold1, P("D3"), 50, 118)
    # drums per Bach bar (= 32 sixteenths)
    for b in range(23):
        t = E0 + b * 32 * S16
        if b < 4:
            continue
        if b < 9:
            A.drums(t, {CH: "o.x.o.x.o.x.o.x.o.x.o.x.o.x.o.x.", K: "x...............x...............",
                        RS: "........o...............o......."}, 0.8)
        elif b < 14:
            pat = {CH: "x.x.x.x.x.x.x.x.x.x.x.x.x.x.x.x.", K: "X...........x...X.....x.........",
                   SN: "........X...............X......."}
            if b == 9:
                pat[CR] = "X"
            if b == 13:
                pat = {CH: "x.x.x.x.x.x.x.x.x.x.x.x.", K: "X...........x...X.....x.",
                       SN: "........X...............xxx.", THI: "........................x..x",
                       TMID: "..........................x.", TLO: "............................xx",
                       K2: "............................X."}
            A.drums(t, pat)
        elif b < 21:
            A.drums(t, {K: "X...X...X...X...X...X...X...X...", OH: "..x...x...x...x...x...x...x...x.",
                        CH: "x...x...x...x...x...x...x...x...", CP: "....x.......x.......x.......x...",
                        CR: "X" if b % 2 == 0 else "", SPL: ("" if b % 2 == 0 else "X")})
        elif b == 21:
            A.drums(t, {K: "X...X...X...X...X...X...X...X...", CH: "x.x.x.x.x.x.x.x.",
                        SN: "....x.......x.....x.x.x.x.x.x.x.", CR: "X"})
        else:  # b == 22: the D7 hold (half a Bach bar = 16 sixteenths)
            A.drums(t, {K: "X...X...X...X...", SN: "xxxxxxxxXXXXXXXX", CR: "............X..."}, 0.95)

    # ============================================================ F: climax
    F0 = bt(90, E0)
    A.marker(F0, "Climax: U.N. Owen main theme + Bach counter-melody on organ")
    A.mix(F0, drums=94, lead=106, dbar=74)
    main_theme(F0, climax=True)

    # ============================================================ G: fugue ending
    G0 = F0 + 16 * 16 * S16
    A.marker(G0, "Fugue ending (BWV 542, m106-115)")
    A.mix(G0, drums=90, lead=96)
    START, FINAL = 420.0, 456.0

    def gt(beat):
        return G0 + (beat - START) * BB

    for n in FU:
        if n["track"] not in (1, 3, 5) or not START <= n["start"] < FINAL + 1:
            continue
        s, d, p = n["start"], n["dur"], n["pitch"]
        final = s >= FINAL
        dd = (4 * BB if final else d * BB) - 24
        t = gt(s)
        if n["track"] == 5:
            A.note("pedal", t, dd, p, 92)
            A.note("pedal", t, dd, p - 12, 76)
            A.note("bass", t, dd - 10, p, 96)
            A.note("gtr", t, dd - 10, p + 12, 84)
            if not final and s % 2 == 0:
                A.note("timp", t, Q, p if p < P("D3") else p - 12, 90)
        else:
            A.note("organ", t, dd, p, 94, human=2)
            if d >= 1.0 or final:
                A.note("choir", t, dd, p, 76)
                A.note("str", t, dd, p + (0 if n["track"] == 1 else 12), 72)
    # the pedal subject (m110 b2.5) gets the square lead on top
    for n in FU:
        if n["track"] == 5 and 437.5 <= n["start"] < 449:
            A.note("lead", gt(n["start"]), n["dur"] * BB - 24, n["pitch"] + 24, 80)
    # final G-major chord: everyone
    tf = gt(FINAL)
    fin = 4 * BB
    A.chord("choir", tf, fin, [P("G2"), P("D3"), P("G3"), P("B3"), P("D4"), P("G4"), P("B4")], 90)
    A.chord("str", tf, fin, [P("G3"), P("B3"), P("D4"), P("G4"), P("B4"), P("D5")], 86)
    A.chord("brass", tf, fin, [P("G3"), P("D4"), P("G4"), P("B4"), P("D5")], 90)
    A.chord("lead", tf, fin - Q, [P("B5"), P("D6"), P("G6")], 86)
    A.chord("dbar", tf, fin - Q, [P("G4"), P("B4"), P("D5")], 74)
    A.note("bass", tf, fin, P("G1"), 100)
    A.chord("gtr", tf, fin - Q, [P("G2"), P("D3"), P("G3")], 92)
    A.chord("hpsi", tf, 2 * Q, [P("G3"), P("B3"), P("D4"), P("G4"), P("B4")], 100)
    A.note("cele", tf, 3 * Q, P("G6"), 80)
    A.note("cele", tf, 3 * Q, P("B6"), 70)
    A.drums(tf, {K: "X", CR: "X", CR2: "X", CHN: "X"})
    A.timp_roll(tf, tf + fin - Q, P("G2"), 118, 40, rate=Q // 8)
    A.ramp("choir", tf, tf + fin, 11, 127, 70)
    A.ramp("str", tf, tf + fin, 11, 127, 60)
    # drums for m106-114
    for b in range(9):
        t = G0 + b * 32 * S16
        if b < 7:
            pat = {K: "X...X...X...X...X...X...X...X...", OH: "..x...x...x...x...x...x...x...x.",
                   CH: "x...x...x...x...x...x...x...x...", CP: "....x.......x.......x.......x..."}
            if b in (0, 2, 6):
                pat[CR] = "X"
            if b == 4:
                pat[CR] = "................X"            # pedal subject enters at m110 b2.5
                pat[CR2] = "................X"
            if b == 3:
                pat[SN] = "........................xx.xXXXX"
            A.drums(t, pat)
        else:  # m113-114: half time, ritardando
            A.drums(t, {K: "X.......X.......X.......X.......", SN: "........X...............X.......",
                        CH: "x.x.x.x.x.x.x.x.x.x.x.x.x.x.x.x.", CR: "X" if b == 7 else "X...............X",
                        TLO: ("" if b == 7 else "........................x.x.xxxx"),
                        TMID: ("" if b == 7 else "........................x.x.....")})
    # ritardando into the final chord
    for beat, bpm in ((448, 150), (450, 144), (452, 136), (453, 128), (454, 118), (455, 102),
                      (455.5, 88), (FINAL, 66)):
        A.tempo(gt(beat), bpm)

    mf = A.save(out_path, "BWV 542 x U.N. Owen Was Her?")
    return A, mf, dict(A=0, B=B0, C=C0, D=D0, E=E0, F=F0, G=G0, end=tf + fin)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    arr, mf, marks = build(*sys.argv[1:])
    print(f"wrote {sys.argv[4]}: {len(arr.notes)} notes, {mf.length:.1f} s")
    for k, v in marks.items():
        print(f"  {k}: tick {v}")
