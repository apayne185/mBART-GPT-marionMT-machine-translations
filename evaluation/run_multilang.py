"""
Multi-language benchmark evaluation using OPUS-100.

Runs all MT models on en→de, en→es, and en→ar to test which models are
most robust across language families:
  - German  (de): Germanic, closely related to English, rich morphology
  - Spanish (es): Romance, moderate syntactic distance from English
  - Arabic  (ar): Semitic, right-to-left, very rich morphology, different script

Key research question: do multilingual models (mBART-50, NLLB-200) hold their
advantage over language-pair-specific models (MarianMT) as linguistic distance
from English increases? NLLB-200 was specifically designed for typologically
diverse and lower-resource language pairs — Arabic is the strongest test of this.

Uses OPUS-100 (not WMT14) so the same dataset format and test split size
applies across all three languages, making cross-language score comparisons fair.

Usage:
    python evaluation/run_multilang.py           # n=100 per language (~30 min CPU)
    python evaluation/run_multilang.py 50        # faster, less stable BLEU
    python evaluation/run_multilang.py 200       # more stable, longer run

Outputs:
    evaluation/multilang_results.csv — scores for all models × languages
"""

import csv
import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from corpus_loader import load_opus100_pairs
from lang_config import LANG_CONFIG
from metrics import evaluate, compute_labse
from model_loaders import build_registry

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
LANGS = list(LANG_CONFIG.keys())  # ["de", "es", "ar"]

print(f"Multi-language benchmark — en → {{de, es, ar}}, n={N} per language")
print(f"Dataset: OPUS-100 (test split)\n")

# all_results[lang][model] = {metric: value}
all_results: dict[str, dict[str, dict]] = {}
# all_timing[lang][model] = seconds
all_timing: dict[str, dict[str, float]] = {}

for lang in LANGS:
    cfg = LANG_CONFIG[lang]
    print("=" * 70)
    print(f"LANGUAGE: English → {cfg['name']} ({lang}), n={N}")
    print("=" * 70)

    print(f"Loading OPUS-100 {cfg['opus_pair']} test split...")
    sources, references = load_opus100_pairs(tgt_lang=lang, n=N)
    print(f"Loaded {len(sources)} pairs.\n")

    registry = build_registry(tgt_lang=lang)
    lang_translations: dict[str, list[str]] = {}
    lang_timing: dict[str, float] = {}
    objects_to_free: list = []

    for name, loader in registry.items():
        print(f"[{name}] Translating {N} sentences...")
        t0 = time.time()
        try:
            translate_fn, objects_to_free = loader()
            translations = []
            for i, src in enumerate(sources):
                translations.append(translate_fn(src))
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{N}", end="\r", flush=True)
            lang_translations[name] = translations
            elapsed = time.time() - t0
            lang_timing[name] = elapsed
            print(f"[{name}] Done ({elapsed:.1f}s)    ")
        except Exception as e:
            print(f"[{name}] Skipped — {e}")
        finally:
            objects_to_free.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Score
    lang_scores: dict[str, dict] = {}
    for name, translations in lang_translations.items():
        s = evaluate(translations, references, lang=cfg["bert_lang"])
        s["LaBSE"] = compute_labse(sources, translations)
        lang_scores[name] = s

    all_results[lang] = lang_scores
    all_timing[lang]  = lang_timing

    # Print per-language table
    col_w = 13
    metric_names = list(next(iter(lang_scores.values())).keys())
    header = f"{'Model':<22}" + "".join(f"{k:>{col_w}}" for k in metric_names)
    print(f"\nResults — en → {lang} ({cfg['name']}):")
    print(header)
    print("-" * len(header))
    for name, s in lang_scores.items():
        print(f"{name:<22}" + "".join(f"{v:>{col_w}.2f}" for v in s.values()))
    print()

# ---------------------------------------------------------------------------
# Cross-language summary
# ---------------------------------------------------------------------------

print("=" * 70)
print("CROSS-LANGUAGE SUMMARY — BLEU and LaBSE per model")
print("=" * 70)

mt_models = list(next(iter(all_results.values())).keys())
col_w = 10

# Header: Model | de BLEU | de LaBSE | es BLEU | es LaBSE | ar BLEU | ar LaBSE
header = f"{'Model':<22}"
for lang in LANGS:
    header += f"  {lang} BLEU{'':{col_w-7}}{lang} LaBSE"
print(header)
print("-" * len(header))

for name in mt_models:
    row = f"{name:<22}"
    for lang in LANGS:
        if name in all_results[lang]:
            s = all_results[lang][name]
            row += f"  {s['BLEU']:>7.2f}   {s['LaBSE']:>7.2f}"
        else:
            row += f"  {'—':>7}   {'—':>7}"
    print(row)

print("\nBest BLEU per language:")
for lang in LANGS:
    scores = all_results[lang]
    best = max(scores, key=lambda n: scores[n]["BLEU"])
    print(f"  en→{lang} ({LANG_CONFIG[lang]['name']}): {best}  "
          f"BLEU {scores[best]['BLEU']:.2f}  LaBSE {scores[best]['LaBSE']:.2f}")

# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

out_dir = os.path.dirname(__file__)
csv_path = os.path.join(out_dir, "multilang_results.csv")
metric_names = list(next(iter(next(iter(all_results.values())).values())).keys())

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Language", "Model", "N"] + metric_names + ["Time (s)"])
    for lang, scores in all_results.items():
        for model, s in scores.items():
            t = round(all_timing[lang].get(model, 0), 1)
            writer.writerow([lang, model, N] + list(s.values()) + [t])

print(f"\nResults saved to {csv_path}")
