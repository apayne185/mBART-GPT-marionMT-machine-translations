# Machine Translation & Cross-lingual Semantic Analysis

Comparative study of neural machine translation (NMT) models, evaluating translation quality and semantic preservation across architectures — from lightweight encoder-decoder models to large instruction-tuned LLMs.

## Research Questions

1. **Do dedicated MT models outperform instruction-tuned LLMs?**
   mBART-50 and NLLB-200 were built specifically for translation; TowerInstruct is a general-purpose LLM fine-tuned for it; GPT-2 is an untuned baseline. Does architectural specialisation still matter when LLMs can be prompted?

2. **Does surface-level evaluation agree with semantic evaluation?**
   BLEU measures word overlap against a reference — a model could score well by copying common words while failing to preserve meaning. BERTScore and LaBSE measure semantic similarity via embeddings. Do they rank models the same way BLEU does?

3. **Are semantic similarity findings robust across embedding models?**
   LaBSE and `paraphrase-multilingual-mpnet-base-v2` are trained independently on different corpora. If both agree on which MT model best preserves meaning, that conclusion is more credible than if only one embedding model said so.

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

Run the full evaluation pipeline (translates with all models, scores, exports CSV and chart):

```bash
python evaluation/run_comparison.py
```

Outputs:
- Console: translation table + scored metrics table
- `evaluation/results.csv` — scores for all models and metrics
- `evaluation/results.png` — grouped bar chart

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
│   ├── data.py            # Shared source sentences, references, and labels
│   ├── model_loaders.py   # Shared model loading functions (used by all pipelines)
│   ├── metrics.py         # BLEU, chrF, METEOR, BERTScore, LaBSE scoring functions
│   ├── run_comparison.py  # Runs all models, scores, exports CSV + chart
│   └── visualize.py       # Grouped bar chart (can run standalone from results.csv)
│   # results.csv and results.png are generated on first run (gitignored)
├── langchain_pipeline/
│   ├── judge.py      # LCEL judge chain: ChatPromptTemplate | ChatHuggingFace | StrOutputParser | RunnableLambda
│   └── pipeline.py   # Full pipeline: translate → LLM judge → rank comparison
│   # judge_results.json is generated on run (gitignored)
├── environment.yml     # Conda environment (includes langchain + langchain-anthropic)
└── requirements.txt    # Pip dependencies
```
