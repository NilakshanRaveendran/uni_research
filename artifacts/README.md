# Analysis artifacts — which generation is authoritative

| Path | Status |
|---|---|
| `features_wideband.csv` | **authoritative** — the feature table the dissertation reports |
| `out_wideband/` | **authoritative** — the analysis outputs the dissertation reports |
| `features_speech.csv` | superseded (60–400 Hz search band); see `out_final/README.md` |
| `out_final/` | superseded; retained only to audit the correction in Chapter 4 |
| `features.csv` | earliest generation; superseded |
| `finetune/` | current — MarianMT fine-tuning histories, comparisons and hypotheses |

The dissertation's numbers regenerate from the authoritative files via
`thesis/scripts/thesis_numbers.py`. The viva answer card regenerates via
`thesis/scripts/make_answer_card.py`.
