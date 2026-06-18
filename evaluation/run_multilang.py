"""
All-pairs multilingual MT evaluation using FLORES-200.

Evaluates all models on every directed language pair drawn from the set
{en, de, es, ar}, covering 12 pairs by default:

  en→de  en→es  en→ar
  de→en  de→es  de→ar
  es→en  es→de  es→ar
  ar→en  ar→de  ar→es

FLORES-200 is used for all pairs because it is a multi-way parallel corpus —
the same 1012 sentences are professionally translated and index-aligned across
all 200 languages. This makes BLEU scores directly comparable across any
language direction, including non-English source.

GPT-2 and TowerInstruct are English-source only and are skipped for other pairs.
MarianMT is attempted for all pairs; if a direct Helsinki-NLP/opus-mt-{src}-{tgt}
model does not exist on HuggingFace, that model is skipped gracefully for the pair.

Usage:
    python evaluation/run_multilang.py                    # all 12 pairs, n=100
    python evaluation/run_multilang.py 50                 # faster, less stable BLEU
    python evaluation/run_multilang.py 100 en-de es-ar    # specific pairs only
    python evaluation/run_multilang.py 50 de-es ar-en     # two pairs, fast

Outputs:
    evaluation/multilang_results.csv        — scores for all models × pairs (gitignored)
    evaluation/translations_{src}-{tgt}.csv — per-pair translation outputs (gitignored)
"""

import csv
import gc
import os
import sys
import time
from itertools import permutations

import torch

sys.path.insert(0, os.path.dirname(__file__))
from corpus_loader import load_flores200_pairs
from lang_config import LANG_CONFIG
from metrics import evaluate, compute_labse
from model_loaders import build_registry

# ---------------------------------------------------------------------------
# Parse arguments: optional N and optional pair filter
# ---------------------------------------------------------------------------

N = 100
PAIR_FILTER: list[tuple[str, str]] | None = None

for arg in sys.argv[1:]:
    if arg.isdigit():
        N = int(arg)
    elif "-" in arg:
        parts = arg.split("-")
        if len(parts) == 2 and all(p in LANG_CONFIG for p in parts):
            if PAIR_FILTER is None:
                PAIR_FILTER = []
            PAIR_FILTER.append((parts[0], parts[1]))
        else:
            raise SystemExit(
                f"Unknown pair {arg!r}. "
                f"Format: src-tgt where src and tgt are in {list(LANG_CONFIG)}."
            )

ALL_LANGS = list(LANG_CONFIG.keys())   # ["en", "de", "es", "ar"]
ALL_PAIRS = [(s, t) for s, t in permutations(ALL_LANGS, 2)]
PAIRS = [(s, t) for s, t in ALL_PAIRS if PAIR_FILTER is None or (s, t) in PAIR_FILTER]

if not PAIRS:
    valid = ", ".join(f"{s}-{t}" for s, t in ALL_PAIRS)
    raise SystemExit(f"No valid pairs matched. Available: {valid}")

print(f"All-pairs MT evaluation — FLORES-200, n={N} per pair")
print(f"Pairs ({len(PAIRS)}): {', '.join(f'{s}→{t}' for s, t in PAIRS)}\n")

# ---------------------------------------------------------------------------
# Run each pair
# ---------------------------------------------------------------------------

# all_results[(src, tgt)][model] = {metric: value}
all_results: dict[tuple, dict] = {}
all_timing:  dict[tuple, dict] = {}

for (src, tgt) in PAIRS:
    src_name = LANG_CONFIG[src]["name"]
    tgt_name = LANG_CONFIG[tgt]["name"]
    print("=" * 70)
    print(f"PAIR: {src_name} ({src}) → {tgt_name} ({tgt}), n={N}")
    print("=" * 70)

    try:
        print(f"Loading corpus {src}→{tgt} (FLORES-200 or OPUS-100 fallback)...")
        sources, references = load_flores200_pairs(src, tgt, n=N)
        print(f"Loaded {len(sources)} pairs.\n")
    except Exception as e:
        print(f"Corpus load failed — skipping pair: {e}\n")
        continue

    registry = build_registry(src_lang=src, tgt_lang=tgt)
    pair_translations: dict[str, list[str]] = {}
    pair_timing: dict[str, float] = {}
    objects_to_free: list = []

    for name, loader in registry.items():
        print(f"[{name}] Translating {N} sentences...")
        t0 = time.time()
        try:
            translate_fn, objects_to_free = loader()
            translations = []
            for i, sentence in enumerate(sources):
                translations.append(translate_fn(sentence))
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{N}", end="\r", flush=True)
            pair_translations[name] = translations
            elapsed = time.time() - t0
            pair_timing[name] = elapsed
            print(f"[{name}] Done ({elapsed:.1f}s)    ")
        except Exception as e:
            print(f"[{name}] Skipped — {e}")
        finally:
            objects_to_free.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Score
    bert_lang = LANG_CONFIG[tgt]["bert_lang"]
    pair_scores: dict[str, dict] = {}
    for name, translations in pair_translations.items():
        scores = evaluate(translations, references, lang=bert_lang)
        scores["LaBSE"] = compute_labse(sources, translations)
        pair_scores[name] = scores

    all_results[(src, tgt)] = pair_scores
    all_timing[(src, tgt)]  = pair_timing

    # Per-pair results table
    col_w = 13
    if pair_scores:
        metric_names = list(next(iter(pair_scores.values())).keys())
        header = f"{'Model':<22}" + "".join(f"{k:>{col_w}}" for k in metric_names)
        print(f"\nResults — {src}→{tgt} ({src_name}→{tgt_name}):")
        print(header)
        print("-" * len(header))
        for name, s in pair_scores.items():
            print(f"{name:<22}" + "".join(f"{v:>{col_w}.2f}" for v in s.values()))

    # Save translations for this pair
    model_names_pair = list(pair_translations.keys())
    trans_path = os.path.join(os.path.dirname(__file__), f"translations_{src}-{tgt}.csv")
    with open(trans_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "Source", "Reference"] + model_names_pair)
        for i, (src_sent, ref) in enumerate(zip(sources, references)):
            row_vals = [pair_translations[m][i] for m in model_names_pair]
            writer.writerow([i + 1, src_sent, ref] + row_vals)
    print(f"Translations saved to {trans_path}\n")

# ---------------------------------------------------------------------------
# Cross-pair BLEU summary matrix
# ---------------------------------------------------------------------------

completed = {k: v for k, v in all_results.items() if v}
if not completed:
    print("No successful translations to summarise.")
    sys.exit(0)

all_model_names = sorted({m for scores in completed.values() for m in scores})
col_w = 9

print("=" * 70)
print("BLEU SCORES — all language pairs")
print("=" * 70)
header = f"{'Pair':<10}" + "".join(f"{m:>{col_w + 2}}" for m in all_model_names)
print(header)
print("-" * len(header))
for (src, tgt), scores in completed.items():
    row = f"{src}→{tgt:<7}"
    for m in all_model_names:
        val = f"{scores[m]['BLEU']:.2f}" if m in scores else "—"
        row += f"{val:>{col_w + 2}}"
    print(row)

print("\nBest model per pair (BLEU):")
for (src, tgt), scores in completed.items():
    best = max(scores, key=lambda m: scores[m]["BLEU"])
    print(
        f"  {src}→{tgt}: {best:<22} BLEU {scores[best]['BLEU']:.2f}"
        f"  LaBSE {scores[best]['LaBSE']:.2f}"
    )

# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------

out_dir = os.path.dirname(__file__)
csv_path = os.path.join(out_dir, "multilang_results.csv")

first_scores = next(iter(next(iter(completed.values())).values()))
metric_names = list(first_scores.keys())

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Pair", "Model", "N"] + metric_names + ["Time (s)"])
    for (src, tgt), scores in completed.items():
        for model, s in scores.items():
            t = round(all_timing[(src, tgt)].get(model, 0), 1)
            writer.writerow([f"{src}-{tgt}", model, N] + list(s.values()) + [t])

print(f"\nResults saved to {csv_path}")
