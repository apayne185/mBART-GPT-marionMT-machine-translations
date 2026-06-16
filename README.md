# Machine Translation & Cross-lingual Semantic Analysis

Comparative study of neural machine translation (NMT) models, evaluating translation quality and semantic preservation across architectures — from lightweight encoder-decoder models to large instruction-tuned LLMs.

## Research Questions

1. **Do dedicated MT models outperform instruction-tuned LLMs?**
   mBART-50 and NLLB-200 were built specifically for translation; TowerInstruct is a general-purpose LLM fine-tuned for it; GPT-2 is an untuned baseline. Does architectural specialisation still matter when LLMs can be prompted?

2. **Does surface-level evaluation agree with semantic evaluation?**
   BLEU measures word overlap against a reference — a model could score well by copying common words while failing to preserve meaning. BERTScore and LaBSE measure semantic similarity via embeddings. Do they rank models the same way BLEU does?

3. **Are semantic similarity findings robust across embedding models?**
   LaBSE and `paraphrase-multilingual-mpnet-base-v2` are trained independently on different corpora. If both agree on which MT model best preserves meaning, that conclusion is more credible than if only one embedding model said so.

## Key Findings

### Evaluation results (en → de, 6 sentences)

| Model | BLEU | chrF | METEOR | BERTScore F1 | LaBSE (en↔de) | Load+Infer |
|---|---|---|---|---|---|---|
| **MarianMT** | **51.97** | **75.15** | **69.42** | **93.19** | 90.02 | 26s |
| mBART-50 | 30.82 | 66.97 | 57.59 | 89.91 | **90.33** | 150s |
| NLLB-200 | 27.81 | 64.26 | 56.28 | 90.12 | 89.96 | **15s** |
| GPT-2 | 0.04 | 4.76 | 0.99 | 46.93 | 27.03 | 68s |

*TowerInstruct-7B requires a CUDA-capable GPU and is excluded from this run.*

### WMT14 newstest2014 benchmark results (n=100)

| Model | BLEU | chrF | METEOR | BERTScore F1 | LaBSE (en↔de) | Time |
|---|---|---|---|---|---|---|
| NLLB-200 | **20.81** | 51.42 | 41.71 | 84.53 | 88.09 | 419s |
| MarianMT | 20.38 | **52.55** | **43.57** | **85.18** | **89.66** | 97s |
| mBART-50 | 18.65 | 50.93 | 39.76 | 84.50 | 89.66 | 914s |
| GPT-2 | 0.05 | 8.49 | 0.99 | 51.32 | 30.08 | 1622s |

*Published WMT14 en→de BLEU for these models is typically 24–30; our lower scores reflect n=100 sampling and single-sentence inference without tuned beam search.*

### Cross-dataset comparison — 6 sentences vs WMT14

| Model | BLEU (6-sent) | BLEU (WMT14) | Δ BLEU | LaBSE (6-sent) | LaBSE (WMT14) | Δ LaBSE |
|---|---|---|---|---|---|---|
| MarianMT | 51.97 | 20.38 | −31.59 | 90.02 | 89.66 | −0.36 |
| mBART-50 | 30.82 | 18.65 | −12.17 | 90.33 | 89.66 | −0.67 |
| NLLB-200 | 27.81 | 20.81 | −7.00 | 89.96 | 88.09 | −1.87 |
| GPT-2 | 0.04 | 0.05 | +0.01 | 27.03 | 30.08 | +3.05 |

**The rankings reverse at scale.** MarianMT's BLEU advantage on 6 simple sentences disappears entirely on WMT14 news text — NLLB-200 now leads. NLLB's BLEU drop (−7) is the smallest of the three MT models, suggesting it is the most robust to domain shift. MarianMT's large drop (−31.59) indicates its 6-sentence score was inflated by the simplicity of the hand-crafted evaluation set. **LaBSE is far more stable than BLEU across datasets** (maximum drift: −1.87 vs −31.59), confirming it measures something more fundamental than surface word matching.

A new finding at scale: **NLLB-200 leads BLEU but has the lowest LaBSE** among MT models (88.09 vs 89.66 for MarianMT/mBART). It produces translations closer to the reference in word choice, but slightly less faithful to the source in semantic content. This is only visible with a large enough evaluation set.

### Q1 — Do dedicated MT models outperform instruction-tuned LLMs?

**Partially answered** — TowerInstruct-7B requires a CUDA GPU (driver update pending) so the LLM-fine-tuned case cannot yet be compared. From the models that did run: **yes, the dedicated MT models outperform the untuned baseline decisively.** MarianMT, a 300M-parameter model trained exclusively for en→de, outperforms every other model on all surface metrics. The generalist models (mBART-50, NLLB-200) score lower despite being larger, because they spread capacity across many language pairs. GPT-2 — an untuned language model — completely fails: BLEU 0.04, LaBSE 27.03. It loops or hallucinates in English rather than translating, confirming that language modelling ability alone does not confer translation ability. The more interesting comparison (dedicated MT model vs instruction-tuned LLM such as TowerInstruct) remains open.

**Speed note:** NLLB-200 is the fastest (15s) despite being a 200-language model. mBART-50 is the slowest (150s), likely due to tokenizer overhead from the SentencePiece/protobuf fallback path on this system.

### Q2 — Does surface-level evaluation agree with semantic evaluation?

**Partially, but with an important caveat.** BLEU ranks MarianMT far ahead of mBART-50 (51.97 vs 30.82 — a 21-point gap). However, LaBSE — which measures cross-lingual meaning preservation directly from source to translation without a reference — ranks them near-equally: 90.02 vs 90.33. mBART-50 actually scores *higher* on LaBSE than MarianMT. Both models preserve meaning at the same level; MarianMT simply chooses words closer to the single human reference, inflating its BLEU score. BERTScore narrows the gap further (93.19 vs 89.91). **Conclusion: BLEU overstates the quality gap between specialised and generalist MT models when only one reference translation is available.**

### Q3 — Are semantic similarity findings robust across embedding models?

**Mostly yes, but with one important exception.** Both LaBSE and mpnet rank models identically: MarianMT ≈ mBART-50 ≈ NLLB-200 >> GPT-2. The overall conclusion is robust. However, the two models diverge sharply on the idiom sentence ("raining cats and dogs"):

- **LaBSE:** 0.84 for all three MT models — penalises the literal translation "Es regnet Katzen und Hunde" because the idiomatic *meaning* (heavy rain) is not fully preserved
- **mpnet:** 0.99 for all three MT models — considers the literal translation near-perfect

This is a genuine limitation: mpnet appears to capture surface-level conceptual overlap (raining → regnet, cats → Katzen) without detecting that the idiomatic meaning differs. LaBSE, optimised specifically for cross-lingual alignment, is more sensitive to this mismatch and is the stricter judge for figurative language. For evaluating idiom translation quality, LaBSE is the more reliable signal.

A secondary divergence: mpnet is more lenient toward GPT-2's English outputs (scores 0.09–0.75) than LaBSE (−0.10–0.48), likely because mpnet was trained on more English-heavy multilingual data.

### Notable observations

- **All three MT models are visually indistinguishable on the LaBSE heatmap.** MarianMT, mBART-50, and NLLB-200 occupy the same colour range (0.84–0.97). The 21-point BLEU gap between MarianMT and NLLB-200 does not appear anywhere in the semantic similarity results.
- **mBART-50 scores highest on mpnet for the "Neural nets" sentence (0.91 vs 0.80 for MarianMT, 0.85 for NLLB-200).** mBART kept more loanwords from English ("Neural Networks", "Repräsentationen") which happen to be closer to the source in mpnet's embedding space — illustrating that higher embedding similarity does not always mean more natural German.
- **Idiom failure (all MT models):** "It's raining cats and dogs" → all models produce the literal "Es regnet Katzen und Hunde" rather than the idiomatic "Es regnet in Strömen". NMT models lack the cultural knowledge to resolve figurative language.
- **Negative cosine similarity (GPT-2, "Neural nets"):** LaBSE scored GPT-2's output at −0.10 where it looped "The following text is from a paper by the same author". Negative cosine similarity means the output points in the *opposite direction* from the source in embedding space — not just wrong, but semantically anti-correlated.
- **Technical terms are easiest:** "XLM-E code" scored highest across all MT models (LaBSE 0.96–0.97) because the proper noun XLM-E requires no translation and anchors the sentence semantically.

## Models

| Model | Architecture | HuggingFace ID | Notes |
|---|---|---|---|
| **MarianMT** | Encoder-decoder | `Helsinki-NLP/opus-mt-en-de` | Lightweight, language-pair-specific |
| **mBART-50** | Encoder-decoder | `facebook/mbart-large-50-many-to-many-mmt` | Multilingual, 50 languages |
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
| **LaBSE (en↔de)** | Cross-lingual embedding similarity from source to translation — reference-free |

## LangChain Integration

An LLM-as-judge evaluation pipeline built with [LangChain LCEL](https://python.langchain.com/docs/concepts/lcel/). It runs after translation to score each model on three linguistic dimensions that automated metrics cannot capture.

### Judge dimensions

| Dimension | What it measures |
|---|---|
| **Fluency** | Grammatical correctness and naturalness in German |
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

- **Rankings agree with BLEU**: MarianMT ≥ mBART-50 ≥ NLLB-200 > GPT-2 in both metrics, adding a third data point in favour of this ordering.
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

Run the full evaluation pipeline on 6 hand-crafted sentences (good for quick inspection and translation output review):

```bash
python evaluation/run_comparison.py
```

Outputs:
- Console: translation table + scored metrics table
- `evaluation/results.csv` — scores for all models and metrics
- `evaluation/results.png` — grouped bar chart

Run the WMT14 benchmark evaluation against a real MT benchmark dataset (newstest2014, 3003 professionally translated en→de sentences — the same test set used to evaluate the original Transformer):

```bash
python evaluation/run_benchmark.py        # first 100 sentences (~10 min on CPU)
python evaluation/run_benchmark.py 500    # larger subset for more stable BLEU
python evaluation/run_benchmark.py 3003   # full test set (~3 hrs on CPU)
```

Outputs:
- Console: corpus-level metrics table + timing
- `evaluation/benchmark_results.csv` — scores for all models and metrics
- `evaluation/benchmark_results.png` — grouped bar chart

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

Different models use different language code conventions:

| Model | English | German |
|---|---|---|
| MarianMT | `en` | `de` (encoded in model name) |
| mBART-50 | `en_XX` | `de_DE` |
| NLLB-200 | `eng_Latn` | `deu_Latn` (FLORES-200 format) |
| TowerInstruct | `English` | `German` (natural language) |

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
│   ├── data.py            # 6 hand-crafted source sentences, references, and labels
│   ├── wmt14_loader.py    # Loads WMT14 newstest2014 from HuggingFace (en→de, up to 3003 pairs)
│   ├── model_loaders.py   # Shared model loading functions (used by all pipelines)
│   ├── metrics.py         # BLEU, chrF, METEOR, BERTScore, LaBSE scoring functions
│   ├── run_comparison.py  # Runs all models on 6 sentences, exports CSV + chart
│   ├── run_benchmark.py   # Runs all models on WMT14 subset, exports CSV + chart
│   └── visualize.py       # Grouped bar chart (can run standalone from any results CSV)
│   # results.csv, results.png, benchmark_results.csv, benchmark_results.png gitignored
├── langchain_pipeline/
│   ├── judge.py      # LCEL judge chain: ChatPromptTemplate | ChatHuggingFace | StrOutputParser | RunnableLambda
│   └── pipeline.py   # Full pipeline: translate → LLM judge → rank comparison
│   # judge_results.json is generated on run (gitignored)
├── environment.yml     # Conda environment (includes langchain + langchain-huggingface)
└── requirements.txt    # Pip dependencies
```
