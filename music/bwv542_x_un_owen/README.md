# BWV 542 × U.N. Owen Was Her?

New melodies fused from two tunes: the fugue subject of J. S. Bach's *Fantasia and Fugue
in G minor* (BWV 542, "Great") and ZUN's *U.N. Owen Was Her?* (Touhou 6). The tunes are not
played on top of each other. Each theme is **one line** that turns from Bach into Owen and
back, with one shared rhythm and one chord progression.

`bwv542_x_un_owen_was_her.mid` is a General MIDI type-1 file with 14 instrument tracks
and section markers. It runs about 2:05.

## The fused themes (G minor, 155 bpm)

Owen is moved from D minor to G minor. Bach's fragments run at half time, so his 16ths
become 8ths and move at the same speed as Owen's line.

- **Theme A (verse).** Bach's opening turn, *D | B♭ C A B♭ G G′*, ends on G. That's
  exactly the note Owen's theme starts on, so the line carries straight on with Owen's
  bars 2–4. The answering phrase starts with Owen's bar 1, puts Bach's *F♯ G E F♯ D G D*
  where Owen's bar 2 was, and finishes with Owen's turn down to G.
- **Theme B (chorus).** Owen's falling-third chromatic line (*D B♭ C♯ A | E C♯ C E♭*)
  runs into Bach's falling-third sequence (*D E♭ C D B♭ E♭ B♭ | C A D D′*). This happens
  twice, then Bach's turn makes a cadence on G.
- **Theme C (bridge).** Bach's rising run climbs into Owen's bars 1–3, then hands over
  to Bach's sequence, which lands on D7.
- **Riff.** Owen's 5/4 intro arpeggio, with every other stab replaced by Bach's turn,
  as one harpsichord line.

Every Bach and Owen bar in the themes is a literal fragment of the source. The script
checks each one against the source MIDI files when it runs. A checker confirms there
are no minor-2nd or minor-9th clashes between the melody and the accompaniment.

| Time | Section |
|------|---------|
| 0:00 | Fused riff (Owen's arpeggio + Bach's turn), building up |
| 0:15 | Theme A on music box (celesta + glockenspiel) |
| 0:28 | Theme A on synth lead in thirds, with the band |
| 0:40 | Theme B, the chorus |
| 0:53 | Fused riff interlude |
| 1:00 | Theme A in D minor (Owen's original key) on organ, as a baroque episode |
| 1:13 | Theme C, the bridge, building up |
| 1:25 | Theme B, climax: choir, guitar, full kit |
| 1:38 | Theme A, finale |
| 1:50 | Bach's own final cadence (m114–115) to G major, with Owen's arpeggio over it |

## Regenerating

```sh
pip install mido
python3 make_mashup.py bwv542-a4-1.mid "15. U.N. Owen was her (ZUN).mid" out.mid
```

- `bwv542-a4-1.mid` is the fugue MIDI from the
  [Mutopia Project edition](https://www.mutopiaproject.org/ftp/BachJS/BWV542/bwv542/)
  (public domain).
- ZUN's MIDI is used only to verify the Owen fragments. It is not bundled here.

*U.N. Owen Was Her?* © ZUN / Team Shanghai Alice. This is a non-commercial fan arrangement.
