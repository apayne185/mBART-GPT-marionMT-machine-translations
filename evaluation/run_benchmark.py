"""
WMT14 Benchmark Evaluation

Runs all MT models against the first N sentences of WMT14 newstest2014
(default N=100) and scores them with BLEU, chrF, METEOR, BERTScore, and
LaBSE. Results are exported to benchmark_results.csv and benchmark_results.png.

This complements run_comparison.py (6 hand-crafted sentences, good for
quick inspection) with a statistically credible corpus-level evaluation.
WMT14 is the standard benchmark used to evaluate the original Transformer
and most subsequent MT research, making results directly comparable to the
published literature.

Usage:
    python evaluation/run_benchmark.py          # 100 sentences (default)
    python evaluation/run_benchmark.py 200      # custom N
    python evaluation/run_benchmark.py 3003     # full test set (~3 hrs on CPU)
"""

import csv
import gc
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from metrics import evaluate, compute_labse
from model_loaders import build_registry
from wmt14_loader import load_wmt14_pairs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100

print(f"Loading WMT14 newstest2014 — first {N} sentence pairs...")
SOURCES, REFERENCES = load_wmt14_pairs(n=N)
print(f"Loaded {len(SOURCES)} pairs.\n")

# ---------------------------------------------------------------------------
# Translate with each model sequentially
# ---------------------------------------------------------------------------

print("=" * 70)
print(f"BENCHMARK TRANSLATION (WMT14 en→de, n={N})")
print("=" * 70)

registry = build_registry()
all_translations: dict[str, list[str]] = {}
timing: dict[str, float] = {}
objects_to_free: list = []

for name, loader in registry.items():
    print(f"\n[{name}] Translating {N} sentences...")
    t0 = time.time()
    try:
        translate_fn, objects_to_free = loader()
        translations = []
        for i, src in enumerate(SOURCES):
            translations.append(translate_fn(src))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{N}", end="\r", flush=True)
        all_translations[name] = translations
        elapsed = time.time() - t0
        timing[name] = elapsed
        print(f"[{name}] Done ({elapsed:.1f}s, {elapsed/N:.2f}s/sentence)    ")
    except Exception as e:
        print(f"[{name}] Skipped — {e}")
    finally:
        objects_to_free.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print(f"EVALUATION METRICS — WMT14 newstest2014, n={N}, en→de")
print("=" * 70)

scores: dict[str, dict] = {}
for name, translations in all_translations.items():
    s = evaluate(translations, REFERENCES, lang="de")
    s["LaBSE (en↔de)"] = compute_labse(SOURCES, translations)
    scores[name] = s

col_w = 14
metric_names = list(next(iter(scores.values())).keys())
header = f"{'Model':<22}" + "".join(f"{k:>{col_w}}" for k in metric_names)
print(header)
print("-" * len(header))
for name, s in scores.items():
    print(f"{name:<22}" + "".join(f"{v:>{col_w}.2f}" for v in s.values()))

print(f"\nTime per model (load + {N} sentences):")
for name, t in timing.items():
    print(f"  {name:<22} {t:.1f}s  ({t/N:.2f}s/sent)")

# ---------------------------------------------------------------------------
# Export: metrics CSV + translations CSV
# ---------------------------------------------------------------------------

out_dir = os.path.dirname(__file__)

# Translations — useful for inspecting what each model produced per sentence
model_names = list(all_translations.keys())
trans_path = os.path.join(out_dir, "benchmark_translations.csv")
with open(trans_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["i", "Source", "Reference"] + model_names)
    for i, (src, ref) in enumerate(zip(SOURCES, REFERENCES)):
        writer.writerow([i + 1, src, ref] + [all_translations[m][i] for m in model_names])
print(f"Translations saved to {trans_path}")

csv_path = os.path.join(out_dir, "benchmark_results.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "N"] + metric_names + ["Time (s)"])
    for name, s in scores.items():
        writer.writerow([name, N] + list(s.values()) + [round(timing.get(name, 0), 1)])
print(f"\nResults saved to {csv_path}")

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

try:
    from visualize import save_chart
    chart_path = os.path.join(out_dir, "benchmark_results.png")
    save_chart(scores, chart_path)
except Exception as e:
    print(f"Visualization skipped: {e}")
