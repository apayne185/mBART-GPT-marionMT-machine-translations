"""
Quick comparison run on 6 hand-crafted sentences.

Good for inspecting translation output visually and catching obvious failures.
Use run_benchmark.py (WMT14) or run_multilang.py (OPUS-100) for statistically
credible corpus-level evaluation.

Usage:
    python evaluation/run_comparison.py          # English → German (default)
    python evaluation/run_comparison.py es       # English → Spanish
    python evaluation/run_comparison.py ar       # English → Arabic
"""

import csv
import gc
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import SOURCES, LABELS, get_references
from lang_config import LANG_CONFIG
from metrics import evaluate, compute_labse
from model_loaders import build_registry

TGT_LANG = sys.argv[1] if len(sys.argv) > 1 else "de"
if TGT_LANG not in LANG_CONFIG:
    raise SystemExit(f"Unknown language {TGT_LANG!r}. Choose from: {list(LANG_CONFIG)}")

REFERENCES = get_references(TGT_LANG)
LANG_NAME  = LANG_CONFIG[TGT_LANG]["name"]
BERT_LANG  = LANG_CONFIG[TGT_LANG]["bert_lang"]

MODEL_REGISTRY = build_registry(TGT_LANG)

# ---------------------------------------------------------------------------
# Translate with each model sequentially, freeing memory between each
# ---------------------------------------------------------------------------

all_translations = {}
timing = {}
objects_to_free = []

for name, loader in MODEL_REGISTRY.items():
    print(f"[{name}] Loading...")
    t0 = time.time()
    try:
        translate_fn, objects_to_free = loader()
        all_translations[name] = [translate_fn(src) for src in SOURCES]
        elapsed = time.time() - t0
        timing[name] = elapsed
        print(f"[{name}] Done ({elapsed:.1f}s)")
    except Exception as e:
        print(f"[{name}] Skipped — {e}")
    finally:
        objects_to_free.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Print translations
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print(f"TRANSLATIONS — en → {LANG_NAME}")
print("=" * 70)
for i, src in enumerate(SOURCES):
    print(f"\nSource [{LABELS[i]}]: {src}")
    print(f"Reference:            {REFERENCES[i]}")
    for name, translations in all_translations.items():
        print(f"  {name:<20} {translations[i]}")

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print(f"EVALUATION METRICS — en → {LANG_NAME}")
print("=" * 70)

scores = {}
for name, translations in all_translations.items():
    s = evaluate(translations, REFERENCES, lang=BERT_LANG)
    s["LaBSE (en↔tgt)"] = compute_labse(SOURCES, translations)
    scores[name] = s

col_w = 14
header = f"{'Model':<22}" + "".join(f"{k:>{col_w}}" for k in next(iter(scores.values())))
print(header)
print("-" * len(header))
for name, s in scores.items():
    print(f"{name:<22}" + "".join(f"{v:>{col_w}.2f}" for v in s.values()))

print("\nTime per model (load + inference):")
for name, t in timing.items():
    print(f"  {name:<22} {t:.1f}s")

# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

csv_path = os.path.join(os.path.dirname(__file__), f"results_{TGT_LANG}.csv")
metric_names = list(next(iter(scores.values())).keys())
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model"] + metric_names + ["Time (s)"])
    for name, s in scores.items():
        writer.writerow([name] + list(s.values()) + [round(timing.get(name, 0), 1)])
print(f"\nResults saved to {csv_path}")

# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

try:
    from visualize import save_chart
    chart_path = os.path.join(os.path.dirname(__file__), f"results_{TGT_LANG}.png")
    save_chart(scores, chart_path)
except Exception as e:
    print(f"Visualization skipped: {e}")
