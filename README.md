# Machine Translation & Cross-lingual Semantic Analysis

Comparative study of neural machine translation (NMT) models, evaluating translation quality and semantic preservation across architectures — from lightweight encoder-decoder models to large instruction-tuned LLMs — across multiple target languages.

## Research Questions

1. **Do dedicated MT models outperform instruction-tuned LLMs?**
   mBART-50 and NLLB-200 were built specifically for translation; TowerInstruct is a general-purpose LLM fine-tuned for it; GPT-2 is an untuned baseline. Does architectural specialisation still matter when LLMs can be prompted?

2. **Does surface-level evaluation agree with semantic evaluation?**
   BLEU measures word overlap against a reference — a model could score well by copying common words while failing to preserve meaning. BERTScore and LaBSE measure semantic similarity via embeddings. Do they rank models the same way BLEU does?

3. **Are semantic similarity findings robust across embedding models?**
   LaBSE and `paraphrase-multilingual-mpnet-base-v2` are trained independently on different corpora. If both agree on which MT model best preserves meaning, that conclusion is more credible than if only one embedding model said so.

4. **Does architectural breadth matter more as linguistic distance from English increases?**
   MarianMT uses a dedicated model per language pair; mBART-50 and NLLB-200 use a single multilingual model. Does the specialist-vs-generalist tradeoff shift as the target language diverges typologically from English?

## Key Findings

### Evaluation results (en → de, 6 sentences)

| Model | BLEU | chrF | METEOR | BERTScore F1 | LaBSE (en↔tgt) | Load+Infer |
|---|---|---|---|---|---|---|
| NLLB-200 | **56.89** | **76.99** | **74.46** | 91.90 | 90.03 | 19s |
| **MarianMT** | 51.97 | 75.15 | 69.42 | **93.19** | 90.02 | 5s |
| mBART-50 | 33.49 | 67.01 | 59.63 | 90.11 | **90.21** | 14s |
| GPT-2 | 0.04 | 4.76 | 0.99 | 46.93 | 27.03 | 39s |

*All dedicated MT models use beam search (num_beams=4). GPT-2 uses greedy decode. TowerInstruct-7B requires a CUDA-capable GPU. Load+Infer times shown are for cached models (first run includes download).*

### WMT14 newstest2014 benchmark results (n=100)

| Model | BLEU | chrF | METEOR | BERTScore F1 | LaBSE (en↔tgt) | Time |
|---|---|---|---|---|---|---|
| NLLB-200 | **20.96** | 52.38 | 43.09 | 84.78 | 88.20 | 503s |
| MarianMT | 20.38 | **52.55** | **43.57** | **85.18** | **89.66** | 50s |
| mBART-50 | 18.94 | 51.05 | 40.04 | 84.51 | 89.71 | 351s |
| GPT-2 | 0.05 | 8.49 | 0.99 | 51.32 | 30.08 | 865s |

*Published WMT14 en→de BLEU is typically 26–28; remaining gap reflects n=100 sampling and sentence-level (non-batched) inference.*

### Cross-dataset comparison — 6 sentences vs WMT14

| Model | BLEU (6-sent) | BLEU (WMT14) | Δ BLEU | LaBSE (6-sent) | LaBSE (WMT14) | Δ LaBSE |
|---|---|---|---|---|---|---|
| MarianMT | 51.97 | 20.38 | −31.59 | 90.02 | 89.66 | −0.36 |
| mBART-50 | 33.49 | 18.94 | −14.55 | 90.21 | 89.71 | −0.50 |
| NLLB-200 | 56.89 | 20.96 | −35.93 | 90.03 | 88.20 | −1.83 |
| GPT-2 | 0.04 | 0.05 | +0.01 | 27.03 | 30.08 | +3.05 |

**BLEU rankings are unstable; LaBSE rankings are not.** All three MT models collapse in BLEU from simple sentences to news text (drops of 14–36 points), but their LaBSE scores barely move (drifts of −0.36 to −1.83). This confirms LaBSE measures something more fundamental than surface word matching. **mBART-50 is the most consistent MT model at scale** — its BLEU drop (−14.55) is less than half of MarianMT's (−31.59) or NLLB-200's (−35.93).

A notable result: **NLLB-200's simple-sentence advantage (56.89 BLEU) largely disappears on WMT14 (20.96)**, matching MarianMT closely. Beam search inflates NLLB's score on short, simple sentences more than on long news text. NLLB still leads on WMT14 BLEU, but the margin is negligible. **LaBSE at scale reveals a hidden trade-off**: NLLB produces translations closest to the reference (highest WMT14 BLEU) but with the lowest LaBSE (88.20 vs 89.66/89.71 for MarianMT/mBART-50) — its translations are more reference-like but slightly less faithful to the source's meaning, a gap that only appears with a large enough evaluation set.

### Multi-language comparison (en→de/es/ar, 6 sentences)

Evaluating on German, Spanish, and Arabic reveals how model architecture interacts with linguistic distance.

| Model | en→de BLEU | en→es BLEU | en→ar BLEU | Architecture |
|---|---|---|---|---|
| MarianMT | 51.97 | **63.53** | 25.75 | Language-pair-specific (separate model per pair) |
| mBART-50 | 33.49 | 54.07 | **51.37** | Multilingual (50 languages, one model) |
| NLLB-200 | **56.89** | 58.95 | 35.01 | Multilingual (200 languages, one model) |
| GPT-2 | 0.04 | 0.07 | 0.07 | Untuned causal LM |

**Ranking by language:**

| Language | 1st | 2nd | 3rd |
|---|---|---|---|
| German | NLLB-200 (56.89) | MarianMT (51.97) | mBART-50 (33.49) |
| Spanish | MarianMT (63.53) | NLLB-200 (58.95) | mBART-50 (54.07) |
| Arabic | **mBART-50 (51.37)** | NLLB-200 (35.01) | **MarianMT (25.75)** |

**The Arabic ranking is a complete inversion of Spanish.** MarianMT leads on Spanish (63.53) but collapses to last on Arabic (25.75). mBART-50, which trails on both German and Spanish, jumps to first on Arabic (51.37) — a 25-point gap over MarianMT. This is the central finding of the multi-language evaluation.

### Q1 — Do dedicated MT models outperform instruction-tuned LLMs?

**Partially answered** — TowerInstruct-7B requires a CUDA GPU (driver update pending) so the LLM-fine-tuned case cannot yet be compared. From the models that did run: **yes, dedicated MT models outperform the untuned baseline decisively.** All three MT models (MarianMT, mBART-50, NLLB-200) score above 50 BLEU on Spanish and German; GPT-2 scores below 0.10 across all three languages. GPT-2 loops or hallucinates in English rather than translating, confirming that language modelling ability alone does not confer translation ability. The more interesting comparison (dedicated MT model vs instruction-tuned LLM such as TowerInstruct) remains open.

The multi-language results add nuance to Q1: the question is not just *dedicated vs general* but *which type of dedicated model*. For closely related languages (German, Spanish), NLLB-200 or MarianMT leads. For Arabic (Semitic, non-Latin script), mBART-50's multilingual training dominates. Architecture matters, but the right architecture depends on the target language.

### Q2 — Does surface-level evaluation agree with semantic evaluation?

**Partially, but with an important caveat.** BLEU places a 23-point gap between NLLB-200 (56.89) and mBART-50 (33.49) on German. However, LaBSE — which measures cross-lingual meaning preservation directly from source to translation, without a reference — ranks them near-equally: 90.03 vs 90.21. mBART-50 actually scores *higher* on LaBSE than NLLB-200 despite its much lower BLEU. Both models preserve meaning at the same level; NLLB-200 and MarianMT simply choose words closer to the single human reference, inflating their BLEU scores. BERTScore narrows the gap further (91.90 vs 90.11). **Conclusion: BLEU overstates the quality gap between MT models when only one reference translation is available.**

### Q3 — Are semantic similarity findings robust across embedding models?

**Mostly yes, but with one important exception.** Both LaBSE and mpnet rank models identically: MarianMT ≈ mBART-50 ≈ NLLB-200 >> GPT-2. The overall conclusion is robust. However, the two models diverge sharply on the idiom sentence ("raining cats and dogs"):

- **LaBSE:** 0.84 for all three MT models — penalises the literal translation "Es regnet Katzen und Hunde" because the idiomatic *meaning* (heavy rain) is not fully preserved
- **mpnet:** 0.99 for all three MT models — considers the literal translation near-perfect

This is a genuine limitation: mpnet appears to capture surface-level conceptual overlap (raining → regnet, cats → Katzen) without detecting that the idiomatic meaning differs. LaBSE, optimised specifically for cross-lingual alignment, is more sensitive to this mismatch and is the stricter judge for figurative language.

### Q4 — Does architectural breadth matter more as linguistic distance increases?

**Yes, dramatically.** MarianMT uses a separate dedicated model per language pair; mBART-50 uses a single model spanning 50 languages. On Spanish (a Romance language, moderate morphological distance from English), the specialist wins: MarianMT BLEU 63.53 vs mBART-50 54.07. On Arabic (Semitic, right-to-left, non-Latin script, very high typological distance), the multilingual model wins decisively: mBART-50 BLEU 51.37 vs MarianMT 25.75.

**Interpretation:** A dedicated en→ar MarianMT model is trained exclusively on English–Arabic data. mBART-50 is trained on 50 languages simultaneously, giving it broader exposure to morphologically rich and typologically distant languages. For Arabic, the cross-linguistic transfer learned from a multilingual corpus outweighs the specificity of a dedicated pair model. NLLB-200 (200 languages) also outperforms MarianMT on Arabic (35.01 vs 25.75), confirming the trend — multilingual breadth becomes increasingly valuable as linguistic distance grows.

### Notable observations

- **All three MT models are visually indistinguishable on the LaBSE heatmap.** MarianMT, mBART-50, and NLLB-200 occupy the same colour range (0.84–0.97). The 23-point BLEU gap between NLLB-200 and mBART-50 on German does not appear anywhere in the semantic similarity results.
- **mBART-50 scores highest on mpnet for the "Neural nets" sentence (0.91 vs 0.80 for MarianMT, 0.85 for NLLB-200).** mBART kept more loanwords from English ("Neural Networks", "Repräsentationen") which happen to be closer to the source in mpnet's embedding space — illustrating that higher embedding similarity does not always mean more natural German.
- **Idiom failure is universal across all three languages.** "It's raining cats and dogs" produces literal translations in German ("Es regnet Katzen und Hunde"), Spanish ("Está lloviendo gatos y perros"), and Arabic ("إنها تمطر القطط والكلاب") across all models. No model produces the idiomatic equivalents ("Es regnet in Strömen", "Está lloviendo a cántaros"). NMT models lack the cultural knowledge to resolve figurative language, regardless of target language.
- **mBART-50 generates a non-existent Spanish word on the idiom sentence.** On "It's raining cats and dogs", mBART-50 produces a word that does not exist in Spanish. This type of hallucination — a fluent-looking but invented word — is a known failure mode of multilingual models whose vocabularies span many languages.
- **Negative cosine similarity (GPT-2, "Neural nets"):** LaBSE scored GPT-2's output at −0.10 where it looped "The following text is from a paper by the same author". Negative cosine similarity means the output points in the *opposite direction* from the source in embedding space — not just wrong, but semantically anti-correlated.
- **Technical terms are easiest:** "XLM-E code" scored highest across all MT models (LaBSE 0.96–0.97) because the proper noun XLM-E requires no translation and anchors the sentence semantically.
- **MarianMT's scores were unchanged by explicitly adding num_beams=4.** Its `generation_config.json` already specifies beam search. This was confirmed by running with and without the parameter — scores are identical.

## Models

| Model | Architecture | HuggingFace ID | Notes |
|---|---|---|---|
| **MarianMT** | Encoder-decoder | `Helsinki-NLP/opus-mt-en-{tgt}` | Lightweight, language-pair-specific; separate model per target language |
| **mBART-50** | Encoder-decoder | `facebook/mbart-large-50-many-to-many-mmt` | Multilingual, 50 languages; one model for all pairs |
| **NLLB-200** | Encoder-decoder | `facebook/nllb-200-distilled-600M` | Meta's successor to mBART, 200 languages |
| **GPT-2** | Decoder-only (causal LM) | `gpt2` | Prompted translation baseline |
| **TowerInstruct-7B** | Decoder-only (instruction-tuned) | `Unbabel/TowerInstruct-7B-v0.2` | LLaMA-2 fine-tuned for MT; requires CUDA |

### Semantic Analysis & Evaluation

| Tool | Purpose |
|---|---|
| **LaBSE** | Cross-lingual cosine similarity between source and translation — measures meaning preservation without needing a reference translation |
| **paraphrase-multilingual-mpnet-base-v2** | Second independent embedding model used to verify that similarity-based model rankings are robust across embedding choices |

### Evaluation Metrics

| Metric | What it measures |
|---|---|
| **BLEU** | Word n-gram precision against reference (surface overlap) |
| **chrF** | Character n-gram overlap — more robust for German morphology |
| **METEOR** | Accounts for synonyms and stemming; better semantic proxy than BLEU |
| **BERTScore F1** | Contextual embedding similarity between hypothesis and reference |
| **LaBSE (en↔tgt)** | Cross-lingual embedding similarity from source to translation — reference-free |

## LangChain Integration

An LLM-as-judge evaluation pipeline built with [LangChain LCEL](https://python.langchain.com/docs/concepts/lcel/). It runs after translation to score each model on three linguistic dimensions that automated metrics cannot capture.

### Judge dimensions

| Dimension | What it measures |
|---|---|
| **Fluency** | Grammatical correctness and naturalness in the target language |
| **Adequacy** | Preservation of source meaning without omissions or distortions |
| **Style** | Appropriateness of register and word choice for a native speaker |

### How it works

1. **Stage 1** — reuses the existing model loaders to collect translations from all MT models (sequentially, memory-safe)
2. **Stage 2** — loads `Qwen/Qwen2.5-1.5B-Instruct` locally via `langchain-huggingface`, constructs an LCEL chain (`ChatPromptTemplate | ChatHuggingFace | StrOutputParser | RunnableLambda`), and evaluates all (source, translation) pairs with `.batch()`
3. **Comparison** — cross-references LLM rankings against corpus-level BLEU to test whether surface metrics agree with LLM judgement (Research Question 2)

### Judge results (Qwen2.5-1.5B-Instruct, en → de)

| Model | Fluency | Adequacy | Style | Overall |
|---|---|---|---|---|
| MarianMT | 8.83 | 7.83 | 7.33 | **7.71** |
| mBART-50 | 8.83 | 7.83 | 7.33 | **7.71** |
| NLLB-200 | 8.83 | 7.83 | 7.33 | **7.71** |
| GPT-2 | 6.83 | 6.33 | 6.33 | 6.44 |

**Key findings from the LLM judge:**

- **Rankings broadly agree with BLEU**: all three MT models outperform GPT-2 in both metrics. BLEU and LLM judge converge on the same bottom-line conclusion: dedicated MT models >> untuned LLM baseline.
- **The three MT models score identically (7.71)**, consistent with the LaBSE finding that they are semantically equivalent. The judge could not discriminate between them — which a larger model (7B+) likely would.
- **The idiom sentence scores highest among MT models (8.2/10)** — the 1.5B judge awards high fluency because "Es regnet Katzen und Hunde" is grammatically correct German, even though it is not the idiomatic phrase. This is a known limitation of smaller LLM judges: they can assess grammar but may miss cultural/idiomatic errors.
- **GPT-2 scores 6.44**, far more generously than BLEU's near-zero. The judge correctly penalises GPT-2's worst outputs ("MT systems": 4.1, "XLM-E code": 3.6) but is too lenient where GPT-2 produced plausible-looking English text.
- **Limitation:** The generic, template-like comments ("The translation is fluent and conveys the full meaning accurately") reflect the capacity ceiling of a 1.5B judge. A 7B or 70B model would produce more discriminating and nuanced evaluation.

### Setup

No API key required — the judge model (`Qwen/Qwen2.5-1.5B-Instruct`) runs locally. It is downloaded automatically from HuggingFace on first run and uses 4-bit quantisation on CUDA or float32 on CPU.

```bash
python langchain_pipeline/pipeline.py
```

Outputs:
- Console: LLM judge scores table, LLM rank vs BLEU rank comparison, per-sentence comments for best and worst model
- `langchain_pipeline/judge_results.json` — full structured output (gitignored)

## Setup

```bash
conda env create -f environment.yml
conda activate nlp-mt
```

## Usage

Run the quick comparison on 6 hand-crafted sentences — good for inspecting translation output and catching obvious failures:

```bash
python evaluation/run_comparison.py          # English → German (default)
python evaluation/run_comparison.py es       # English → Spanish
python evaluation/run_comparison.py ar       # English → Arabic
```

Outputs per language:
- Console: translation table + scored metrics table
- `evaluation/results_{lang}.csv` — scores for all models and metrics (gitignored)
- `evaluation/results_{lang}.png` — grouped bar chart (gitignored)
- `evaluation/translations_{lang}.csv` — what each model produced per sentence (gitignored)

Run the all-pairs evaluation on FLORES-200 (multi-way parallel corpus, any src→tgt):

```bash
python evaluation/run_multilang.py                    # all 12 pairs, n=100 per pair
python evaluation/run_multilang.py 50                 # faster run, less stable BLEU
python evaluation/run_multilang.py 100 en-de es-ar    # specific pairs only
python evaluation/run_multilang.py 50 de-es ar-en     # two pairs, quick test
```

Covers all 12 directed pairs from {en, de, es, ar}: en↔de, en↔es, en↔ar, de↔es, de↔ar, es↔ar. GPT-2 and TowerInstruct are skipped for non-English source. MarianMT is skipped for any pair where a direct `Helsinki-NLP/opus-mt-{src}-{tgt}` model does not exist.

Outputs:
- Console: per-pair metric tables + BLEU summary matrix across all pairs
- `evaluation/multilang_results.csv` — all models × all pairs (gitignored)
- `evaluation/translations_{src}-{tgt}.csv` — per-pair translation outputs (gitignored)

After running, generate the full results document:

```bash
python evaluation/generate_results_md.py
```

This reads `multilang_results.csv` and writes **[RESULTS.md](RESULTS.md)** — full BLEU, chrF, METEOR, BERTScore, and LaBSE tables for all 12 pairs with per-metric winner annotations. Re-run after any new evaluation to refresh it. `RESULTS.md` is gitignored (generated artifact).

Run the WMT14 benchmark against a standard MT research dataset (newstest2014, 3003 professionally translated en→de sentences — the same test set used to evaluate the original Transformer):

```bash
python evaluation/run_benchmark.py          # first 100 sentences (~10 min on CPU)
python evaluation/run_benchmark.py 500      # larger subset for more stable BLEU
python evaluation/run_benchmark.py 3003     # full test set (~3 hrs on CPU)
```

Outputs:
- Console: corpus-level metrics table + timing
- `evaluation/benchmark_results.csv` — scores for all models and metrics (gitignored)
- `evaluation/benchmark_results.png` — grouped bar chart (gitignored)
- `evaluation/benchmark_translations.csv` — translations for offline review (gitignored)

Run cross-lingual semantic similarity analysis across all models (embeds source and translations using two independent embedding models, outputs per-sentence table and heatmap):

```bash
python semantic_analysis/semantic_similarity.py
```

Outputs:
- Console: per-sentence similarity scores + embedding model agreement table
- `semantic_analysis/similarity_heatmap.png` — heatmap (models × sentences, coloured by cosine similarity)

Regenerate the evaluation chart from a saved CSV without re-running models:

```bash
python evaluation/visualize.py
```

Run any model individually:

```bash
python marionMT/marionMT-model.py
python mbart/mbart-model.py
python nllb200/nllb200-model.py
python gpt/gpt-model.py
python towerinstruct/towerinstruct-model.py  # requires CUDA GPU
python labse/labse-model.py
```

## Language Codes

Different model families use different language code conventions:

| Model | English | German | Spanish | Arabic |
|---|---|---|---|---|
| MarianMT | `en` | `de` (in model name) | `es` (in model name) | `ar` (in model name) |
| mBART-50 | `en_XX` | `de_DE` | `es_XX` | `ar_AR` |
| NLLB-200 | `eng_Latn` | `deu_Latn` | `spa_Latn` | `arb_Arab` (FLORES-200) |
| TowerInstruct | `English` | `German` | `Spanish` | `Arabic` (natural language) |

All language-specific codes are centralised in `evaluation/lang_config.py`. To add a new language, add one entry there.

## Project Structure

```
├── marionMT/           # MarianMT encoder-decoder model
├── mbart/              # mBART-50 multilingual model
├── nllb200/            # NLLB-200 distilled model
├── gpt/                # GPT-2 prompted translation baseline
├── towerinstruct/      # TowerInstruct-7B instruction-tuned LLM
├── labse/              # LaBSE standalone demo
├── semantic_analysis/
│   └── semantic_similarity.py  # Per-sentence similarity analysis with LaBSE + mpnet
│   # similarity_heatmap.png is generated on run (gitignored)
├── evaluation/
│   ├── lang_config.py     # Central language config: model codes for de/es/ar across all families
│   ├── data.py            # 6 hand-crafted source sentences, references (de/es/ar), and labels
│   ├── corpus_loader.py   # Loads OPUS-100 test pairs for any supported language
│   ├── wmt14_loader.py    # Loads WMT14 newstest2014 from HuggingFace (en→de, up to 3003 pairs)
│   ├── model_loaders.py   # Shared model loading functions; accepts tgt_lang for multi-language
│   ├── metrics.py         # BLEU, chrF, METEOR, BERTScore, LaBSE scoring functions
│   ├── run_comparison.py  # 6 sentences × any language; exports CSV + chart + translations
│   ├── run_multilang.py   # FLORES-200 evaluation across all 12 directed pairs; exports BLEU matrix
│   ├── run_benchmark.py        # WMT14 benchmark (en→de); exports CSV + chart + translations
│   ├── generate_results_md.py  # Reads multilang_results.csv → writes RESULTS.md
│   └── visualize.py            # Grouped bar chart (can run standalone from any results CSV)
│   # results_*.csv, results_*.png, translations_*.csv, benchmark_*, multilang_results.csv gitignored
├── langchain_pipeline/
│   ├── judge.py      # LCEL judge chain: ChatPromptTemplate | ChatHuggingFace | StrOutputParser | RunnableLambda
│   └── pipeline.py   # Full pipeline: translate → LLM judge → rank comparison
│   # judge_results.json is generated on run (gitignored)
├── environment.yml     # Conda environment (includes langchain + langchain-huggingface)
└── requirements.txt    # Pip dependencies
```
