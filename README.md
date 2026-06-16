# Machine Translation & Cross-lingual Semantic Analysis

Comparative study of neural machine translation (NMT) models, evaluating translation quality and semantic preservation across architectures — from lightweight encoder-decoder models to large instruction-tuned LLMs.

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

### Evaluation Metrics

| Metric | What it measures |
|---|---|
| **BLEU** | Word n-gram precision against reference (surface overlap) |
| **chrF** | Character n-gram overlap — more robust for German morphology |
| **METEOR** | Accounts for synonyms and stemming; better semantic proxy than BLEU |
| **BERTScore F1** | Contextual embedding similarity between hypothesis and reference |
| **LaBSE (en↔de)** | Cross-lingual embedding similarity from source to translation — reference-free |

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

Regenerate the chart from a saved CSV without re-running models:

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
├── labse/              # LaBSE cross-lingual semantic similarity
├── evaluation/
│   ├── metrics.py      # BLEU, chrF, METEOR, BERTScore, LaBSE scoring functions
│   ├── run_comparison.py  # Runs all models, scores, exports CSV + chart
│   ├── visualize.py    # Grouped bar chart (can run standalone from results.csv)
│   ├── results.csv     # Generated output — scores per model
│   └── results.png     # Generated output — bar chart
├── environment.yml     # Conda environment
└── requirements.txt    # Pip dependencies
```
