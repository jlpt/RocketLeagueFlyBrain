# BWV 542 × U.N. Owen Was Her?

A MIDI mashup of J. S. Bach's *Fantasia and Fugue in G minor* (BWV 542, "Great")
and ZUN's *U.N. Owen Was Her?* (Touhou 6, Flandre Scarlet's theme).

`bwv542_x_un_owen_was_her.mid` is a General MIDI type-1 file with 13 instrument tracks
and section markers. It runs about 3:22 and sounds best with a decent GM soundfont,
for example FluidR3_GM.

| Time | Section |
|------|---------|
| 0:00 | **Fantasia** (BWV 542, bars 1–3) on organ and choir. It ends on F♯°7 over a G pedal, which resolves into… |
| 0:17 | **Owen's intro riff** in 5/4, built up on harpsichord, celesta, organ stabs and drums |
| 0:32 | **Owen's chromatic theme** over planing organ chords (Gm–F♯m–A–A♭) |
| 0:45 | **Owen's main theme**. In its second half the harpsichord plays a counter-melody built from the fugue subject's cells (B♭ C A B♭ G G′ / F♯ G E F♯ D G D) |
| 1:09 | **Fugue exposition** (m1–22) at half time (♩≈77, so Bach's 16ths match Owen's 8ths). The band comes in voice by voice: the lead doubles the tenor entry, and guitar and bass double the pedal entry. It ends on a D7 half cadence… |
| 2:19 | **Climax**: …which resolves deceptively (V→VI) into Owen's main theme on E♭, with the Bach counter-melody now on organ, plus choir, guitar and the full kit |
| 2:44 | **Fugue ending** (m106–115), including the last pedal entry of the subject, a ritardando, and Bach's G-major final chord |

Owen is transposed from D minor down to G minor so both pieces share Bach's key.

## Regenerating

```sh
pip install mido
python3 make_mashup.py bwv542.mid bwv542-a4-1.mid "15. U.N. Owen was her (ZUN).mid" out.mid
```

- `bwv542.mid` and `bwv542-a4-1.mid` are the Fantasia and Fugue MIDIs from the
  [Mutopia Project edition](https://www.mutopiaproject.org/ftp/BachJS/BWV542/bwv542/)
  (public domain).
- ZUN's MIDI is used only as a reference for the melody, bass and harmony. It is
  not bundled here. Everything is re-orchestrated.

*U.N. Owen Was Her?* © ZUN / Team Shanghai Alice. This is a non-commercial fan arrangement.
