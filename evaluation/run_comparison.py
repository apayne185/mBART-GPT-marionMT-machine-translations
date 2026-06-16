import csv
import gc
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from metrics import evaluate, compute_labse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Reference data — 6 sentences covering simple, technical, idiomatic, and
# formal registers. Swap in a FLORES-200 subset here for larger-scale eval.
# ---------------------------------------------------------------------------
SOURCES = [
    "The weather is beautiful today.",
    "Machine translation systems have improved significantly over the past decade.",
    "Neural networks learn representations from data using gradient descent.",
    "It's raining cats and dogs.",
    "I cannot find any example code for XLM-E.",
    "The conference on natural language processing attracted researchers from around the world.",
]

REFERENCES = [
    "Das Wetter ist heute wunderbar.",
    "Maschinelle Übersetzungssysteme haben sich im vergangenen Jahrzehnt erheblich verbessert.",
    "Neuronale Netze lernen Repräsentationen aus Daten mithilfe des Gradientenabstiegs.",
    "Es regnet in Strömen.",
    "Ich kann keinen Beispielcode für XLM-E finden.",
    "Die Konferenz über natürliche Sprachverarbeitung zog Forscher aus aller Welt an.",
]

# ---------------------------------------------------------------------------
# Model loaders — each returns (translate_fn, [objects_to_free])
# Loading/unloading sequentially keeps peak memory low.
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
                  tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).items()}
        with torch.no_grad():
            output = model.generate(**inputs, forced_bos_token_id=forced_bos)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_nllb():
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
    tokenizer = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
    forced_bos = tokenizer.convert_tokens_to_ids("deu_Latn")

    def translate(text):
        tokenizer.src_lang = "eng_Latn"
        inputs = {k: v.to(device) for k, v in
                  tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).items()}
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
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
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
# Run
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "MarianMT":   load_marianmt,
    "mBART-50":   load_mbart,
    "NLLB-200":   load_nllb,
    "GPT-2":      load_gpt2,
}

if torch.cuda.is_available():
    MODEL_REGISTRY["TowerInstruct-7B"] = load_towerinstruct
else:
    print("No CUDA detected — skipping TowerInstruct-7B.\n")

all_translations = {}
timing = {}

for name, loader in MODEL_REGISTRY.items():
    print(f"[{name}] Loading...")
    t0 = time.time()
    try:
        translate_fn, objects_to_free = loader()
        translations = [translate_fn(src) for src in SOURCES]
        elapsed = time.time() - t0
        all_translations[name] = translations
        timing[name] = elapsed
        print(f"[{name}] Done ({elapsed:.1f}s)")
    except Exception as e:
        print(f"[{name}] Skipped — {e}")
    finally:
        try:
            for obj in objects_to_free:
                del obj
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Evaluate and print results
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("TRANSLATIONS")
print("=" * 70)
for i, src in enumerate(SOURCES):
    print(f"\nSource: {src}")
    print(f"Reference: {REFERENCES[i]}")
    for name, translations in all_translations.items():
        print(f"  {name:<20} {translations[i]}")

print("\n" + "=" * 70)
print("EVALUATION METRICS (corpus-level, en→de)")
print("=" * 70)

scores = {}
for name, translations in all_translations.items():
    s = evaluate(translations, REFERENCES, lang="de")
    # LaBSE cross-lingual: measures meaning preservation from source to translation
    # without needing a reference — complementary to reference-based metrics above
    s["LaBSE (en↔de)"] = compute_labse(SOURCES, translations)
    scores[name] = s

col_w = 14
header = f"{'Model':<22}" + "".join(f"{k:>{col_w}}" for k in next(iter(scores.values())))
print(header)
print("-" * len(header))
for name, s in scores.items():
    row = f"{name:<22}" + "".join(f"{v:>{col_w}.2f}" for v in s.values())
    print(row)

print("\nTime per model (load + inference):")
for name, t in timing.items():
    print(f"  {name:<22} {t:.1f}s")

# ---------------------------------------------------------------------------
# Export results to CSV
# ---------------------------------------------------------------------------
csv_path = os.path.join(os.path.dirname(__file__), "results.csv")
metric_names = list(next(iter(scores.values())).keys())
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model"] + metric_names + ["Time (s)"])
    for name, s in scores.items():
        writer.writerow([name] + list(s.values()) + [round(timing[name], 1)])
print(f"\nResults saved to {csv_path}")

# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------
try:
    from visualize import save_chart
    chart_path = os.path.join(os.path.dirname(__file__), "results.png")
    save_chart(scores, chart_path)
except Exception as e:
    print(f"Visualization skipped: {e}")
