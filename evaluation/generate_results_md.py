"""
Generate RESULTS.md from the CSV produced by run_multilang.py.

Reads evaluation/multilang_results.csv and writes a formatted markdown file
with BLEU, chrF, METEOR, BERTScore, and LaBSE tables for all language pairs.
Re-run this script after any new evaluation run to refresh RESULTS.md.

Usage:
    python evaluation/generate_results_md.py
"""

import csv
import os
from collections import defaultdict

CSV_PATH = os.path.join(os.path.dirname(__file__), "multilang_results.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "RESULTS.md")

PAIR_ORDER = [
    "en-de", "en-es", "en-ar",
    "de-en", "de-es", "de-ar",
    "es-en", "es-de", "es-ar",
    "ar-en", "ar-de", "ar-es",
]
MODEL_ORDER = ["MarianMT", "mBART-50", "NLLB-200", "GPT-2", "TowerInstruct-7B"]

# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"{CSV_PATH} not found. Run `python evaluation/run_multilang.py` first."
    )

results: dict[str, dict[str, dict]] = defaultdict(dict)
n_val = "?"
metric_names: list[str] = []

with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pair  = row["Pair"]
        model = row["Model"]
        n_val = row["N"]
        if not metric_names:
            metric_names = [k for k in row if k not in ("Pair", "Model", "N", "Time (s)")]
        results[pair][model] = {m: float(row[m]) for m in metric_names}

ordered_pairs  = [p for p in PAIR_ORDER if p in results]
present_models = {m for scores in results.values() for m in scores}
ordered_models = [m for m in MODEL_ORDER if m in present_models]

if not ordered_pairs:
    raise ValueError("No results found in CSV. Check that run_multilang.py completed.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def arrow(pair: str) -> str:
    src, tgt = pair.split("-")
    return f"{src}→{tgt}"

def best_for(pair: str, metric: str) -> tuple[str, float]:
    scores = results[pair]
    best = max(scores, key=lambda m: scores[m].get(metric, -1))
    return best, scores[best].get(metric, 0.0)

def metric_table(metric: str) -> str:
    col_names = ordered_models + ["Best"]
    header = "| Pair | " + " | ".join(col_names) + " |"
    sep    = "|" + "------|" * (len(col_names) + 1)
    rows   = [header, sep]
    for pair in ordered_pairs:
        scores  = results[pair]
        best_m, _ = best_for(pair, metric)
        cells = []
        for m in ordered_models:
            if m in scores:
                val  = scores[m][metric]
                cell = f"**{val:.2f}**" if m == best_m else f"{val:.2f}"
            else:
                cell = "—"
            cells.append(cell)
        rows.append(f"| {arrow(pair)} | {' | '.join(cells)} | {best_m} |")
    return "\n".join(rows)

# ---------------------------------------------------------------------------
# Build markdown
# ---------------------------------------------------------------------------

winner_rows = ""
for pair in ordered_pairs:
    best_b, bleu_v = best_for(pair, "BLEU")
    best_l, lbs_v  = best_for(pair, "LaBSE")
    winner_rows += f"| {arrow(pair)} | {best_b} | {bleu_v:.2f} | {best_l} | {lbs_v:.2f} |\n"

comet_section = ""
if "COMET" in metric_names:
    comet_section = f"\n## COMET (Unbabel/wmt22-comet-da)\n\n{metric_table('COMET')}\n"

md = f"""\
# Machine Translation — Full Results

Generated from `evaluation/multilang_results.csv` by `evaluation/generate_results_md.py`.
Re-run the generator after any new evaluation to refresh these tables.

## Methodology

| Setting | Value |
|---------|-------|
| Dataset | FLORES-200 devtest (where available); OPUS-100 test split for en↔X pairs |
| Sentences per pair | {n_val} |
| Language pairs | All 12 directed combinations from {{en, de, es, ar}} |
| Decoding (MT models) | Beam search — num\\_beams=4, max\\_new\\_tokens=256 |
| Decoding (GPT-2) | Greedy — causal LM baseline, English source only |
| GPT-2 / TowerInstruct | Skipped for non-English source (English-only prompts) |
| MarianMT coverage | Language-pair-specific; pairs with no direct HuggingFace model shown as — |

## BLEU scores

{metric_table("BLEU")}

## chrF scores

{metric_table("chrF")}

## METEOR scores

{metric_table("METEOR")}

## BERTScore F1

{metric_table("BERTScore F1")}

## LaBSE (source ↔ translation)

{metric_table("LaBSE")}
{comet_section}
## Winner summary

| Pair | Best (BLEU) | BLEU | Best (LaBSE) | LaBSE |
|------|-------------|------|--------------|-------|
{winner_rows.rstrip()}

## Notes

- **BLEU** is computed at the sentence level (averaged). Corpus-level BLEU
  (`run_benchmark.py` on WMT14) is lower and more standard for published comparisons.
- **LaBSE** measures cross-lingual semantic similarity between source and translation
  without a reference — it is reference-free and robust to domain shift.
- **FLORES-200** sentences are sourced from English Wikipedia and Wikinews and
  professionally translated into all 200 languages. Every language direction shares
  the same {n_val} sentences, making cross-pair score comparisons fair.
- **Asymmetric pairs** (e.g. en→de vs de→en) are expected to score differently:
  back-translation is a distinct task from forward translation, and training data
  volumes differ by direction.

For architecture details, research questions, and findings see [README.md](README.md).
"""

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(md)

print(f"Written: {os.path.abspath(OUT_PATH)}")
print(f"  Pairs:  {len(ordered_pairs)}")
print(f"  Models: {', '.join(ordered_models)}")
