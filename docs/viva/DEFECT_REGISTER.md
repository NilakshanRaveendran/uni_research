# Defect register — submitted dissertation, CST 498-6

> **For your own preparation. Do not print this for the room.**
>
> The dissertation and the journal manuscript were submitted on 27 August 2026 and cannot be
> changed. This register lists every defect I know of in the submitted text, with the correct value
> and where it comes from, so that nothing is discovered for the first time in the viva.
>
> Every "correct" value below regenerates from `thesis/numbers.json` and
> `artifacts/out_wideband/T4_results.csv` at commit `ea7542c`. Verified against the submitted
> sources, which are preserved in `docs/submitted/` and tagged `submitted-2026-08`.
>
> **The numbers in Chapter 4 are correct.** What is wrong is confined to the abstract, one
> arithmetic sentence, some aggregation wording, and ten last-digit roundings.

## Root cause of most of it

On 27 August the F0 measurement was corrected (commits `312b42d`, `ea7542c`): correlations had been
computed in hertz with no plausibility gate. Chapter 4 and Chapter 5 were regenerated against the
corrected analysis (`artifacts/out_wideband/`). **The abstract was not re-checked**, so it remains a
coherent summary of the superseded analysis (`artifacts/out_final/`). Defects 1–4 all follow from
that single omission.

---

## Tier 1 — will be found, must be pre-empted

### D1. The abstract inverts the thesis's own significance conclusion 🔴

| | |
|---|---|
| Printed | abstract, p. vii: "no prosody result survives family-wise correction" |
| Correct | §4.7.2: three of six *p*-values fall below 0.05/6 = 0.0083, and the same three survive Holm and Benjamini–Hochberg |
| Nuance | Only **two** are findings — pitch range, −5.14 % and −3.89 %. The third (en-es pitch *level*, *p* = 0.0040) has an effect of **−0.02 %** and a CI of ≈ [−0.0003, −0.0001] semitones. λ selected 10⁶, so the model collapsed into the baseline it is compared against. |

**Answer:** §4.7.2 is correct. Three of six clear Bonferroni; two are real — pitch range in both
directions, 5.14 and 3.89 percent. The third is pitch level at *p* = 0.004 with an effect of minus
0.02 percent, which I report as a null so it cannot be counted as a third success. The abstract is
wrong: it was written against the analysis as it stood before I corrected the pitch measurement, and
I updated Chapter 4 without re-checking it.

### D2. Human pitch-level ceiling 🔴

Printed `r = 0.787` (abstract). Correct **0.8488** (Table 4.4, `numbers.json`, RQ2 on p. 41).
Same root cause as D1: 0.787 was computed in hertz with no plausibility gate. The gap reported
throughout the body is **0.311 / 0.314 against 0.849**.

### D3. Ridge improvement 🔴

Printed "9.5 % and 14.7 %" (abstract). Correct **5.14 % and 3.89 %** (Table 4.5, §5.2).
The 9.54/14.65 pair comes from the superseded `artifacts/out_final/T4_results.csv`.

### D4. The gate arithmetic does not add up 🔴 — *first results page*

| | |
|---|---|
| Printed | §4.1, p. 23: "the quality gate … removed 589 pairs, and restriction to single-speaker pairs removed a further 49, leaving 2,304" |
| Problem | **2893 − 589 − 49 = 2255**, not 2304. The 49 is subtracted twice. |
| Verified truth | **540** pairs fail the quality gate. Of the **2,353** survivors, **49** are cross-speaker. 2893 − 540 − 49 = **2304**. And 540 + 49 = **589**, which is the total drop and the 20.4 % quoted (589/2893 = 0.2036). |
| Also reconciles | Table 3.1's 59 cross-speaker pairs: 10 of the 59 also fail the gate, leaving 49 that pass it. |

`numbers.json` stores `dropped_quality_gate = 589`, which conflates the two causes — that is what the
sentence inherited.

**Answer:** You're right, and the sentence is wrong. The gate removes 540. Of the 2,353 that survive
it, 49 are pairs whose two sides carry different speaker identifiers, and those are excluded by the
research question, not by quality. 540 plus 49 is 589, which is the 20.4 percent I quote. The
sentence labels 589 as the gate figure and then subtracts the 49 a second time. Retained is 2,304,
and every downstream number uses 2,304.

### D5. "1.52× longer on average" is a ratio of means 🔴

Printed in the abstract, §4.9, §4.11, §5.5 and journal §IV-F.

| Statistic | Value |
|---|---|
| ratio of pooled means (4.07 / 2.67) | **1.522** ← what is printed |
| **median** per-utterance ratio | **1.425** |
| mean of per-utterance ratios | **1.843** |
| share exceeding 2.0× | **25.1 %** |
| per-direction medians | 1.510 (en-es), 1.325 (es-en) |

**The ten-second kill:** §4.9 puts the pooled 1.52× directly beside subgroup medians of 1.51 and
1.32. **A pooled figure cannot exceed both subgroups.**

**Answer:** It can't, and it doesn't — those are two different statistics sitting in one paragraph,
which is my fault for putting them there. 1.52 is a ratio of means over both directions; 1.51 and
1.32 are medians of per-utterance ratios within each direction. The pooled median is 1.425, which
does sit between them. Figure 4.8 prints the median on the axis. The median is the honest headline.

### D6. Ten double-rounded table cells

Tables 4.5 and B.1. Mechanism verified exactly: true **1.016463** → the four-decimal
`T4_results.md` digest **1.0165** → re-rounded to **1.017**, when correct 3-dp is **1.016**.
Confirmed cells include 1.375→1.374, 1.017→1.016, 0.182→0.181, 0.187→0.186.

**No underlying value is wrong and no conclusion changes.** But it is forensic proof that §3.9's
and §5.4's "no number is typed by hand" is false.

**Answer:** That claim is too strong and I'd withdraw it. The script computes every number and writes
`numbers.json`, but it emits no LaTeX — I transcribed the tables from its four-decimal digest, then
rounded to three. Ten cells read one in the last place high, and all ten are explained by that
double rounding. No value is wrong; the process claim is. I have the exact list.

---

## Tier 2 — answer if asked

### D7. Duration comparison mixes pooled and within-speaker
Abstract compares within-speaker system values (0.352–0.610) against the **pooled** human 0.870.
The within-speaker human figure is **0.8665**.

### D8. "roughly 13× real time" (§4.9.1)
It is the mean of five per-job ratios (13.92), dominated by two very short clips where model loading
dominates. Measured: **3.09, 2.97, 4.44×** on the three clips over a minute; **22.95, 36.14×** on two
15-second clips. Total elapsed over total media = **6.40×**.

**Answer:** It's the mean of five per-job ratios, and it's a bad summary. The three clips over a
minute ran at 3.0, 3.0 and 4.4 times real time. Two fifteen-second clips ran at 23 and 36, because
loading Whisper, MarianMT and XTTS costs the same regardless of clip length. Total elapsed over total
media is 6.4 times. I'd quote 3 to 4 times for real video.

### D9. §4.10 quotes three different quantities as one
- `0.048 → 0.368 → 0.820` — the **mean** F0 statistic on the pre-remeasure `results/synthesis_test.csv`. Reproduces exactly.
- band sweep `0.371/0.589, 0.778/0.739, 0.888/0.856, 0.822/0.819` — the **median** statistic (documented in `prosody.py:12-18`). **Not reproducible from any committed file**, because `PROSODY_COLUMNS` (`pipeline.py:54-60`) never persists `f0_median`.
- Table 4.4 `0.698 / 0.754` — the **mean** statistic on the post-remeasure `results/synthesis_final.csv`. Reproduces exactly.

**Answer:** 0.698 is the reported result. The three are different runs. 0.048, 0.368 and 0.820 are
the same file measured three ways, isolating each defect. 0.698 is the final 65-to-1000-hertz
measurement. The band sweep uses the per-utterance median instead of the mean; it's documented in
`prosody.py`, but I never persisted a median column, so that table is the one thing I cannot
regenerate.

### D10. "Every number regenerates from committed files" (§3.9, §5.4)
~89 numerals in the `.tex` are hand-typed. **They are correct** — the claim about the process is what
is overstated. Genuine exceptions: the dubbing table, the band sweep, the per-speaker ceiling range.

### D11. "roughly 357 usable pairs" (§3.6.3)
Correct **348** (347 in es-en). 357 is the count under the superseded 60–400 Hz feature table — the
narrower band lost voicing on some recordings, so a different set passed the gate.

### D12. "2,304 pairs and 92 speakers" (§4.5)
**94** distinct speaker identifiers appear in the 2,304 retained pairs; **92** contribute more than
one pair and therefore enter the within-speaker decomposition. 92 is the right *n* for that column
and the wrong label for the corpus.

### D13. Duplicate "References" in the table of contents
`main.tex` adds `\addcontentsline{toc}{chapter}{References}` while `apacite` adds its own, so the
built ToC lists **References twice on page 45**.

### D14. Design limitations (in print already, but be ready)
14 test speakers; one contributes a single pair of 348, the largest 78 (22.4 %). In the synthesis
split one speaker contributes 236 of 870 rows (27.1 %). **No listening study.** No external baseline.
No impostor floor for the ECAPA cosine — so "identity is preserved" is a relative, not absolute, claim.

---

## Assets — use these, they are strong

### A1. The *p*-values encode a sign count, not an asymptotic approximation 🟢

For a two-sided Wilcoxon signed-rank test on n = 14, the **smallest attainable** *p* is
2/2¹⁴ = **0.0001220703125**. Verified with scipy:

| Result | *p* | Meaning |
|---|---|---|
| es-en pitch range | **0.0001220703125** | the test's **floor** — attainable only if **all 14 speakers improved** |
| en-es pitch range | **0.001220703125** | exactly **one** speaker moved the wrong way, holding the 5th-smallest magnitude |

**Answer to "n = 14, what power do you have?":** Barely any in the large-sample sense, which is why
I'd rather you read the *p*-value structurally. On fourteen pairs the smallest two-sided signed-rank
*p* is 0.000122, and that is exactly what the Spanish-to-English pitch-range result returns — it
means all fourteen speakers improved. The English-to-Spanish *p* of 0.00122 means exactly one moved
the wrong way, and it held the fifth-smallest change. That is a sign test with fourteen speakers.

### A2. The duration blowup is a synthesis-rate problem, not a translation-length problem 🟢

Measured on `results/synthesis_final.csv`:

| | en-es | es-en |
|---|---|---|
| MT verbosity (MT words / human-target words) | mean **1.056**, median **1.000** | mean **0.990**, median **0.900** |
| XTTS speaking rate vs human | 1.77 vs 3.03 w/s → **1.71× slow** | 2.32 vs 3.42 w/s → **1.48× slow** |

MarianMT's output length is already correct. **XTTS is what is slow.** And XTTS already exposes the
lever: `speed: float = 1.0` at `TTS/tts/models/xtts.py:462`, applied as
`length_scale = 1.0 / max(speed, 0.05)`. `LocalModels.synthesize` never passes it.

This **corrects §4.9.1's stated cause and §5.5's top future-work item**, both of which point at
length-controlled MT. Volunteering that is a demonstration of exactly the disposition §4.10 claims as
a contribution: the plausible fix was the wrong fix, and only measuring revealed it.

### A3. The corpus is what makes the measurement possible
Every correlation is against **the same speaker's real recording in the other language**. That is why
a cosine of 0.42 becomes interpretable. Neither CVSS nor Common Voice can supply it — CVSS's target
speech is TTS-synthesised, and Common Voice's English and Spanish recordings come from different
people. §3.2.1 already says this.

---

## 🔴 Never do this

**Never present the F0 metric correction (0.048 → 0.820) as a model improvement.** It was a
measurement fix, honestly written up in §4.10. Presenting it as a gain is the single most likely way
this defence fails.

**Never study from `docs/viva/PART4_examiner_questions.md` or `FINAL_REVIEW.md`.** Both rehearse the
superseded analysis; PART4 quotes "9.54" twelve times and its multiple-comparisons answer is
inverted. Both are bannered. Study from `docs/viva/ANSWER_CARD.md`.

**Do not put `plots/viva/G*.png` or `plots/synthesis/*` on a slide.** They are pre-correction. Only
`thesis/figures/` is safe.
