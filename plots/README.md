# ⛔ Plot provenance — read before putting any image on a slide

The F0 measurement was corrected on 27 August 2026 (commits `312b42d`, `ea7542c`). Figures built
before that carry the **pre-correction** prosody numbers and will contradict the submitted
dissertation if they appear in a presentation.

| Directory | Built | Status |
|---|---|---|
| `plots/viva/` (15 files, G1–G15) | **26 Aug 12:39** | ⛔ **pre-correction — do not use** |
| `plots/synthesis/` (6 files) | **26 Aug 08:46** | ⛔ **pre-correction — do not use** |
| `thesis/figures/` (19 files) | 27 Aug 02:50 | ✅ safe — these are the submitted figures |
| `journal/figures/` (5 files) | 27 Aug 03:01 | ✅ safe — byte-identical to the thesis PDFs |

Specifically, `G1_prosody_transfer_gap.png` and `G15_dotplot_within_between_decomposition.png` show
the superseded human correlations (0.927 pooled / 0.787 within) against a generated *r* near 0.09 —
the metrics artefact that Section 4.10 of the dissertation exists to correct.

**Build every slide asset from `thesis/figures/` only.** Those are the images the panel has already
seen in the submitted document, so the deck and the dissertation cannot disagree.
