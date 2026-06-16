import csv
import gc
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import SOURCES, REFERENCES
from metrics import evaluate, compute_labse
from model_loaders import build_registry

MODEL_REGISTRY = build_registry()

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
# Evaluate and print results
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("TRANSLATIONS")
print("=" * 70)
for i, src in enumerate(SOURCES):
    print(f"\nSource:    {src}")
    print(f"Reference: {REFERENCES[i]}")
    for name, translations in all_translations.items():
        print(f"  {name:<20} {translations[i]}")

print("\n" + "=" * 70)
print("EVALUATION METRICS (corpus-level, en→de)")
print("=" * 70)

scores = {}
for name, translations in all_translations.items():
    s = evaluate(translations, REFERENCES, lang="de")
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
