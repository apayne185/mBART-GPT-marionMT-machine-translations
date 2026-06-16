"""
Cross-lingual Semantic Similarity Analysis

Embeds source sentences (English) and their translations (German) into a
shared multilingual space using two independent embedding models:
  - LaBSE (Google, 109 languages)
  - paraphrase-multilingual-mpnet-base-v2 (SBERT, 50+ languages)

Cosine similarity between source and translation embeddings measures how
much *meaning* was preserved — independent of surface form. This is the
key distinction from BLEU/chrF, which only measure word/character overlap.

Using two embedding models also lets us check whether model rankings are
robust: if LaBSE and mpnet agree on which MT model preserves meaning best,
that finding is more credible than if only one embedding model said so.
"""

import gc
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works on headless servers with no display
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Sentences — same set as evaluation/run_comparison.py for comparability
# ---------------------------------------------------------------------------

SOURCES = [
    "The weather is beautiful today.",
    "Machine translation systems have improved significantly over the past decade.",
    "Neural networks learn representations from data using gradient descent.",
    "It's raining cats and dogs.",
    "I cannot find any example code for XLM-E.",
    "The conference on natural language processing attracted researchers from around the world.",
]

# Short labels for heatmap axes
LABELS = [
    "Weather",
    "MT systems",
    "Neural nets",
    "Idiom (rain)",
    "XLM-E code",
    "NLP conference",
]

# ---------------------------------------------------------------------------
# Model loaders — load, translate all sentences, free memory
# ---------------------------------------------------------------------------

def load_marianmt():
    from transformers import MarianMTModel, MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(device)

    def translate(text):
        inputs = {k: v.to(device) for k, v in
                  tokenizer(text, return_tensors="pt", padding=True, truncation=True).items()}
        with torch.no_grad():
            output = model.generate(**inputs)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_mbart():
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    ).to(device)
    forced_bos = tokenizer.lang_code_to_id["de_DE"]

    def translate(text):
        tokenizer.src_lang = "en_XX"
        inputs = {k: v.to(device) for k, v in
                  tokenizer(text, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).items()}
        with torch.no_grad():
            output = model.generate(**inputs, forced_bos_token_id=forced_bos)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_nllb():
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
    tokenizer = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    ).to(device)
    forced_bos = tokenizer.convert_tokens_to_ids("deu_Latn")

    def translate(text):
        tokenizer.src_lang = "eng_Latn"
        inputs = {k: v.to(device) for k, v in
                  tokenizer(text, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).items()}
        with torch.no_grad():
            output = model.generate(**inputs, forced_bos_token_id=forced_bos)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_gpt2():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.config.pad_token_id = tokenizer.eos_token_id

    def translate(text):
        prompt = f"Translate the following text from English to German: {text}"
        inputs = {k: v.to(device) for k, v in
                  tokenizer(prompt, return_tensors="pt", padding=True).items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_towerinstruct():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-7B-v0.2")
    model = AutoModelForCausalLM.from_pretrained(
        "Unbabel/TowerInstruct-7B-v0.2",
        quantization_config=quantization_config,
        device_map="auto",
    )

    def translate(text):
        prompt = (
            f"<s>[INST] Translate the following text from English into German.\n"
            f"English: {text}\nGerman: [/INST]"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    return translate, [model, tokenizer]


# ---------------------------------------------------------------------------
# Collect translations
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "MarianMT":  load_marianmt,
    "mBART-50":  load_mbart,
    "NLLB-200":  load_nllb,
    "GPT-2":     load_gpt2,
}

if torch.cuda.is_available():
    MODEL_REGISTRY["TowerInstruct-7B"] = load_towerinstruct
else:
    print("No CUDA detected — skipping TowerInstruct-7B.\n")

all_translations = {}
objects_to_free = []

for name, loader in MODEL_REGISTRY.items():
    print(f"[{name}] Translating...")
    try:
        translate_fn, objects_to_free = loader()
        all_translations[name] = [translate_fn(src) for src in SOURCES]
        print(f"[{name}] Done")
    except Exception as e:
        print(f"[{name}] Skipped — {e}")
    finally:
        objects_to_free.clear()  # drop list references; gc handles the rest
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Semantic similarity with two independent embedding models
# ---------------------------------------------------------------------------

EMBEDDING_MODELS = {
    "LaBSE":   "LaBSE",
    "mpnet":   "paraphrase-multilingual-mpnet-base-v2",
}

# similarities[emb_model][mt_model] = [sim_sentence_0, ..., sim_sentence_n]
similarities = {}

for emb_name, emb_id in EMBEDDING_MODELS.items():
    print(f"\nEmbedding with {emb_name}...")
    emb_model = SentenceTransformer(emb_id)
    src_embs = emb_model.encode(SOURCES, normalize_embeddings=True)

    similarities[emb_name] = {}
    for mt_name, translations in all_translations.items():
        tgt_embs = emb_model.encode(translations, normalize_embeddings=True)
        sims = [float(np.dot(s, t)) for s, t in zip(src_embs, tgt_embs)]
        similarities[emb_name][mt_name] = sims

    del emb_model
    gc.collect()

# ---------------------------------------------------------------------------
# Print per-sentence similarity tables
# ---------------------------------------------------------------------------

mt_models = list(all_translations.keys())
col_w = 12

for emb_name, mt_scores in similarities.items():
    print(f"\n{'=' * 70}")
    print(f"SEMANTIC SIMILARITY PER SENTENCE — {emb_name} (en → de)")
    print(f"{'=' * 70}")
    header = f"{'Sentence':<20}" + "".join(f"{m:>{col_w}}" for m in mt_models)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(LABELS):
        row = f"{label:<20}" + "".join(
            f"{mt_scores[m][i]:>{col_w}.4f}" for m in mt_models
        )
        print(row)
    print("-" * len(header))
    avgs = {m: np.mean(mt_scores[m]) for m in mt_models}
    avg_row = f"{'AVERAGE':<20}" + "".join(f"{avgs[m]:>{col_w}.4f}" for m in mt_models)
    print(avg_row)

# ---------------------------------------------------------------------------
# Embedding model agreement — do LaBSE and mpnet rank models the same way?
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("EMBEDDING MODEL AGREEMENT (average similarity per MT model)")
print(f"{'=' * 70}")
header = f"{'MT Model':<22}{'LaBSE':>{col_w}}{'mpnet':>{col_w}}{'Δ':>{col_w}}"
print(header)
print("-" * len(header))
for mt_name in mt_models:
    labse_avg = np.mean(similarities["LaBSE"][mt_name])
    mpnet_avg = np.mean(similarities["mpnet"][mt_name])
    delta = labse_avg - mpnet_avg
    print(f"{mt_name:<22}{labse_avg:>{col_w}.4f}{mpnet_avg:>{col_w}.4f}{delta:>{col_w}.4f}")

# ---------------------------------------------------------------------------
# Heatmap visualization
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(mt_models) * 0.9 + 2)))

for ax, (emb_name, mt_scores) in zip(axes, similarities.items()):
    data = np.array([mt_scores[m] for m in mt_models])  # (n_models, n_sentences)

    im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(mt_models)))
    ax.set_yticklabels(mt_models, fontsize=10)
    ax.set_title(f"{emb_name}", fontsize=12, fontweight="bold")

    for i in range(len(mt_models)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{data[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if 0.6 < data[i, j] < 0.9 else "white")

    plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)

fig.suptitle(
    "Cross-lingual Semantic Similarity: Source (en) → Translation (de)\n"
    "Higher = more meaning preserved from source to translation",
    fontsize=12,
)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "similarity_heatmap.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nHeatmap saved to {out_path}")
