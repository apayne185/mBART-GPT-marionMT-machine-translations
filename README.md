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

### Semantic Analysis

| Tool | Purpose |
|---|---|
| **LaBSE** | Cross-lingual semantic similarity — measures meaning preservation between source and translation via cosine similarity in a shared 109-language embedding space |

## Setup

```bash
conda env create -f environment.yml
conda activate nlp-mt
```

## Usage

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
├── environment.yml     # Conda environment
└── requirements.txt    # Pip dependencies
```
