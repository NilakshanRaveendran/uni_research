# ⛔ SUPERSEDED ANALYSIS — do not cite, do not reproduce numbers from this directory

This directory holds the prosody analysis as it stood at commit `944df6e`, **before** the
fundamental-frequency measurement was corrected. It is retained only so that the correction reported
in Chapter 4 of the dissertation ("A Correction to the Measurement Itself") can be audited.

**The dissertation reports `artifacts/out_wideband/`, not this directory.**

Two defects were present here, both described in the dissertation:

1. **F0 was correlated in hertz with no plausibility gate.** Pitch perception is logarithmic, so a
   correlation in hertz is dominated by the values furthest from the mean, and a single
   octave-doubled observation is enough to destroy it.
2. **The pitch tracker's search band was narrowed to 60–400 Hz**, which suppressed pyin's voicing
   detection. Measured, that was the worst of the four bands tried. The shipped configuration
   separates a wide *search* band (65–1000 Hz) from a narrower *plausibility* gate (60–400 Hz).

Numbers that differ between the two directories, and which therefore appear in stale documents:

| Quantity | Here (superseded) | `out_wideband/` (reported) |
|---|---|---|
| ridge ΔMAE vs B2, F0 range | −9.54 % / −14.65 % | **−5.14 % / −3.89 %** |
| speaker-clustered *p*, F0 range | 0.0419 / 0.0166 | **0.0012 / 0.0001** |
| learned weight on source F0 range | −0.6034 / −0.5916 | **−0.1906 / −0.1429** |
| test pairs after the quality gate | 357 | **348** (347 in es-en) |
| survives family-wise correction | none | **both pitch-range results** |

`artifacts/features_speech.csv` is the feature table belonging to this generation and is superseded
in the same way. The current feature table is `artifacts/features_wideband.csv`.

If you are preparing for the viva, read `docs/viva/ANSWER_CARD.md`, which is generated from
`thesis/numbers.json`.
