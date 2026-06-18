# Machine Translation — Full Results

Generated from `evaluation/multilang_results.csv` by `evaluation/generate_results_md.py`.
Re-run the generator after any new evaluation to refresh these tables.

## Methodology

| Setting | Value |
|---------|-------|
| Dataset | FLORES-200 devtest (where available); OPUS-100 test split for en↔X pairs |
| Sentences per pair | 100 |
| Language pairs | 6 directed pairs involving English (en↔de, en↔es, en↔ar); 6 cross-lingual pairs require FLORES-200 access |
| Decoding (MT models) | Beam search — num\_beams=4, max\_new\_tokens=256 |
| Decoding (GPT-2) | Greedy — causal LM baseline, English source only |
| GPT-2 / TowerInstruct | Skipped for non-English source (English-only prompts) |
| MarianMT coverage | Language-pair-specific; pairs with no direct HuggingFace model shown as — |

## BLEU scores

| Pair | MarianMT | mBART-50 | NLLB-200 | GPT-2 | Best |
|------|------|------|------|------|------|
| en→de | **34.77** | 31.71 | 28.12 | 0.48 | MarianMT |
| en→es | **33.71** | 17.32 | 31.87 | 0.25 | MarianMT |
| en→ar | **18.46** | 10.56 | 13.60 | 0.03 | MarianMT |
| de→en | **37.95** | 33.39 | 31.59 | — | MarianMT |
| es→en | **40.56** | 18.76 | 38.36 | — | MarianMT |
| ar→en | **33.82** | 19.88 | 30.76 | — | MarianMT |

## chrF scores

| Pair | MarianMT | mBART-50 | NLLB-200 | GPT-2 | Best |
|------|------|------|------|------|------|
| en→de | **55.23** | 53.37 | 50.94 | 7.85 | MarianMT |
| en→es | **57.77** | 40.48 | 57.11 | 6.72 | MarianMT |
| en→ar | **47.87** | 37.43 | 43.62 | 0.15 | MarianMT |
| de→en | **59.16** | 56.04 | 53.17 | — | MarianMT |
| es→en | **62.04** | 42.06 | 58.98 | — | MarianMT |
| ar→en | **60.43** | 47.69 | 57.70 | — | MarianMT |

## METEOR scores

| Pair | MarianMT | mBART-50 | NLLB-200 | GPT-2 | Best |
|------|------|------|------|------|------|
| en→de | **43.32** | 40.94 | 38.77 | 2.39 | MarianMT |
| en→es | **47.34** | 36.87 | 45.63 | 0.98 | MarianMT |
| en→ar | **24.40** | 18.17 | 21.97 | 0.09 | MarianMT |
| de→en | **46.39** | 43.12 | 43.06 | — | MarianMT |
| es→en | **52.91** | 36.23 | 49.66 | — | MarianMT |
| ar→en | **50.31** | 35.04 | 46.05 | — | MarianMT |

## BERTScore F1

| Pair | MarianMT | mBART-50 | NLLB-200 | GPT-2 | Best |
|------|------|------|------|------|------|
| en→de | 84.62 | **84.99** | 83.47 | 51.26 | mBART-50 |
| en→es | **86.47** | 80.12 | 85.87 | 51.47 | MarianMT |
| en→ar | **81.32** | 78.98 | 80.86 | 48.29 | MarianMT |
| de→en | **93.62** | 93.32 | 92.28 | — | MarianMT |
| es→en | **94.89** | 91.68 | 94.05 | — | MarianMT |
| ar→en | **94.93** | 92.77 | 93.14 | — | MarianMT |

## LaBSE (source ↔ translation)

| Pair | MarianMT | mBART-50 | NLLB-200 | GPT-2 | Best |
|------|------|------|------|------|------|
| en→de | 89.35 | **92.22** | 88.40 | 35.19 | mBART-50 |
| en→es | **90.38** | 77.94 | 88.52 | 29.05 | MarianMT |
| en→ar | 83.57 | **89.28** | 83.62 | 23.52 | mBART-50 |
| de→en | 91.70 | **92.41** | 85.45 | — | mBART-50 |
| es→en | **91.23** | 78.36 | 88.02 | — | MarianMT |
| ar→en | **82.42** | 78.38 | 80.50 | — | MarianMT |

## Winner summary

| Pair | Best (BLEU) | BLEU | Best (LaBSE) | LaBSE |
|------|-------------|------|--------------|-------|
| en→de | MarianMT | 34.77 | mBART-50 | 92.22 |
| en→es | MarianMT | 33.71 | MarianMT | 90.38 |
| en→ar | MarianMT | 18.46 | mBART-50 | 89.28 |
| de→en | MarianMT | 37.95 | mBART-50 | 92.41 |
| es→en | MarianMT | 40.56 | MarianMT | 91.23 |
| ar→en | MarianMT | 33.82 | MarianMT | 82.42 |

## Notes

- **BLEU** is computed at the sentence level (averaged). Corpus-level BLEU
  (`run_benchmark.py` on WMT14) is lower and more standard for published comparisons.
- **LaBSE** measures cross-lingual semantic similarity between source and translation
  without a reference — it is reference-free and robust to domain shift.
- **OPUS-100** is used for all 6 pairs here (English-involving only). It is English-centric
  so sentence content differs between pairs — direct cross-pair BLEU comparison should be
  treated as approximate. FLORES-200 (same sentences across all 200 languages) would allow
  fully fair comparison; request access at huggingface.co/datasets/facebook/flores.
- **Asymmetric pairs** (e.g. en→de vs de→en) are expected to score differently:
  back-translation is a distinct task from forward translation, and training data
  volumes differ by direction.

For architecture details, research questions, and findings see [README.md](README.md).
