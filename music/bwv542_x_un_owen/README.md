# BWV 542 × U.N. Owen Was Her?

A MIDI quodlibet: the fugue subject of J. S. Bach's *Fantasia and Fugue in G minor*
(BWV 542, "Great") and the main theme of ZUN's *U.N. Owen Was Her?* (Touhou 6)
**playing at the same time**, over a new chord progression that fits both.

`bwv542_x_un_owen_was_her.mid` is a General MIDI type-1 file with 14 instrument
tracks and section markers. It runs about 2:05.

## How the two tunes fit together

- Owen is moved from D minor to G minor (Bach's key) and stays at its original 155 bpm.
  The fugue plays at half time (♩≈77), so one Bach 16th equals one Owen 8th. Owen's
  8-bar main theme then lasts exactly as long as the fugue subject.
- Owen comes in half a beat after the subject's opening D. Under them runs one chord
  per Bach beat: Gm Gm | D7♯5 C/D | Cm F | B♭ E♭ | D7sus4 D | Gm C7 | C9 F | Gm.
  That's Bach's circle-of-fifths sequence plus Owen's Dorian IV chord.
- Every note of both tunes lands on a chord tone or a short passing note. A checker
  confirms there are no minor-2nd or minor-9th clashes between any two parts, down to
  single 16ths. Pad and harmony notes that would rub against a passing note are dropped.

| Time | Section |
|------|---------|
| 0:00 | Owen's 5/4 intro riff, with the fugue subject's opening (D \| B♭ C A B♭ G G′) laid into it on organ |
| 0:15 | **Both tunes:** Bach's subject on organ, Owen on glockenspiel and celesta |
| 0:28 | **Both tunes in D minor** (Owen's original key): Bach's fugue *answer* on organ, Owen on square lead |
| 0:40 | **Both tunes:** Bach's subject as the bass line, Owen's theme on top |
| 0:53 | Riff reprise, again with the subject's opening |
| 1:00 | Owen's chromatic theme against the subject's opening, repeated over Owen's chords |
| 1:13 | **Both tunes, instruments swapped:** Bach on the synth lead, Owen on the organ |
| 1:25 | **Both tunes, climax:** organ and guitar on Bach, lead on Owen, choir, full kit |
| 1:38 | **Both tunes, finale:** Bach's subject in the pedal and bass under Owen's theme |
| 1:50 | Bach's own final cadence (m114–115) to G major, with Owen's arpeggio over the last chord |

## Regenerating

```sh
pip install mido
python3 make_mashup.py bwv542-a4-1.mid "15. U.N. Owen was her (ZUN).mid" out.mid
```

- `bwv542-a4-1.mid` is the fugue MIDI from the
  [Mutopia Project edition](https://www.mutopiaproject.org/ftp/BachJS/BWV542/bwv542/)
  (public domain).
- ZUN's MIDI is used only as a reference for Owen's melody and riff. It is not bundled
  here. All the harmony and orchestration is new.

*U.N. Owen Was Her?* © ZUN / Team Shanghai Alice. This is a non-commercial fan arrangement.
