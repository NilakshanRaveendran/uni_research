# DRAL data provenance and validation

- Source: `https://www.cs.utep.edu/nigel/dral/DRAL-16kHz.tgz`
- Downloaded: 2026-08-24
- Reported license: CC0 / public domain
- Official archive size: 11,024,258,624 bytes
- Manifest source: `DRAL16kHz/metadata/fragments-short.csv`
- Transcript sources:
  - `https://www.cs.utep.edu/topi/EN_transcripts.txt`
  - `https://www.cs.utep.edu/topi/ES_transcripts.txt`

## Local validation

- 2,893 metadata-linked English/Spanish short-fragment pairs resolved.
- 5,786 short WAV files opened successfully with SoundFile; total duration 4.266 hours.
- 5,784 short WAV files are 16 kHz.
- `EN_072_7.wav` and `ES_072_7.wav` are 44.1 kHz despite the release name. The
  pipeline resamples for feature extraction/ECAPA without modifying raw data.
- 2,834 pairs have matching unique-speaker IDs; 59 have different linked speaker IDs.
- Official transcripts provide 2,871 non-empty English and 2,881 non-empty Spanish
  references. Empty references remain missing for WER/BLEU rather than being imputed.
- `ES_transcripts.txt` line 393 lacks one pipe delimiter. The parser recovers the
  two-character annotator code and emits a warning without changing the raw file.

## Archive packaging note

The outer official archive extracted successfully and contains a second
`DRAL16kHz/dral-16kHz.tgz`. That inner archive includes its own archive pathname and
ends with a truncated-gzip error. The required `fragments-short` audio and metadata
are nevertheless complete and validate as described above. Do not use the inner
archive as an integrity source.
