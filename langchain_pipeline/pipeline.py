"""
LangChain Translation Evaluation Pipeline

Two-stage pipeline:

  Stage 1 — Translation
      Loads all MT models from evaluation/model_loaders.py sequentially,
      translating each source sentence and freeing GPU memory between models.
      This reuses the existing memory-safe infrastructure.

  Stage 2 — LLM-as-Judge (LangChain)
      Loads a local Qwen2.5-1.5B-Instruct model via langchain-huggingface and
      constructs an LCEL chain (see judge.py):
          ChatPromptTemplate | ChatHuggingFace | StrOutputParser | RunnableLambda(extract_json)
      Runs all (source, translation) pairs through the chain using .batch(),
      then frees the judge model before printing results.

  Stage 3 — Comparison
      Cross-references LLM judge rankings against corpus-level BLEU to
      investigate Research Question 2: do surface-level metrics agree with
      LLM judgement when ranking MT models?

Usage:
    conda activate nlp-mt
    python langchain_pipeline/pipeline.py

No API key required — the judge model runs locally.
"""

import gc
import json
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
from data import SOURCES, REFERENCES, LABELS
from metrics import compute_bleu
from model_loaders import build_registry

sys.path.insert(0, os.path.dirname(__file__))
from judge import build_judge_chain


# ---------------------------------------------------------------------------
# Stage 1: Translate with each MT model sequentially
# ---------------------------------------------------------------------------

print("=" * 70)
print("STAGE 1: TRANSLATION")
print("=" * 70)

registry = build_registry()
all_translations: dict[str, list[str]] = {}
objects_to_free: list = []

for name, loader in registry.items():
    print(f"[{name}] Translating...")
    try:
        translate_fn, objects_to_free = loader()
        all_translations[name] = [translate_fn(src) for src in SOURCES]
        print(f"[{name}] Done")
    except Exception as e:
        print(f"[{name}] Skipped — {e}")
    finally:
        objects_to_free.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

mt_models = list(all_translations.keys())

if not mt_models:
    raise RuntimeError("No models produced translations. Aborting.")


# ---------------------------------------------------------------------------
# Stage 2: LLM-as-judge via LangChain LCEL
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("STAGE 2: LLM-AS-JUDGE (local Qwen2.5-1.5B-Instruct via LangChain)")
print("=" * 70)

judge_chain, judge_resources = build_judge_chain()

# Flatten all (source, translation) pairs into a single list for .batch().
# For a local model inference is sequential, but .batch() is still the correct
# LangChain idiom and keeps the code consistent with API-backed workflows.
batch_inputs: list[dict] = []
batch_keys: list[tuple[str, int]] = []  # (model_name, sentence_index)

for model_name, translations in all_translations.items():
    for i, (src, trl) in enumerate(zip(SOURCES, translations)):
        batch_inputs.append({"source": src, "translation": trl})
        batch_keys.append((model_name, i))

print(f"Evaluating {len(batch_inputs)} (source, translation) pairs...")
raw_results = judge_chain.batch(batch_inputs, return_exceptions=True)

# Free the judge model immediately — results are now in raw_results
judge_resources.clear()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("Judge model freed.")

# Reshape flat results into judge_scores[model_name][sentence_index]
judge_scores: dict[str, dict[int, dict]] = {name: {} for name in mt_models}
failed = 0
for (model_name, i), result in zip(batch_keys, raw_results):
    if isinstance(result, Exception):
        print(f"  [Warning] {model_name} sentence {i}: {result}")
        judge_scores[model_name][i] = {
            "fluency": 0, "adequacy": 0, "style": 0, "overall": 0.0,
            "comment": "evaluation failed",
        }
        failed += 1
    else:
        judge_scores[model_name][i] = result

if failed:
    print(f"  {failed}/{len(batch_inputs)} judgement(s) failed — scored 0.")


# ---------------------------------------------------------------------------
# Print: per-model LLM judge averages
# ---------------------------------------------------------------------------

DIMS = ["fluency", "adequacy", "style", "overall"]
col_w = 11
n = len(SOURCES)

print(f"\n{'=' * 70}")
print("LLM-AS-JUDGE SCORES (averaged across all sentences, scored 1–10)")
print("=" * 70)

header = f"{'Model':<22}" + "".join(f"{d:>{col_w}}" for d in DIMS)
print(header)
print("-" * len(header))

judge_averages: dict[str, dict[str, float]] = {}
for name in mt_models:
    avgs = {
        d: sum(judge_scores[name][i].get(d, 0) for i in range(n)) / n
        for d in DIMS
    }
    judge_averages[name] = avgs
    print(f"{name:<22}" + "".join(f"{avgs[d]:>{col_w}.2f}" for d in DIMS))


# ---------------------------------------------------------------------------
# Print: LLM ranking vs BLEU ranking
# ---------------------------------------------------------------------------

print(f"\n{'=' * 70}")
print("LLM JUDGE vs BLEU — does surface-level scoring agree with LLM judgement?")
print("=" * 70)

bleu_scores = {
    name: compute_bleu(translations, REFERENCES)
    for name, translations in all_translations.items()
}

llm_ranked = sorted(mt_models, key=lambda n: judge_averages[n]["overall"], reverse=True)
bleu_ranked = sorted(mt_models, key=lambda n: bleu_scores[n], reverse=True)

header = (
    f"{'Model':<22}"
    f"{'LLM Overall':>13}"
    f"{'LLM Rank':>10}"
    f"{'BLEU':>8}"
    f"{'BLEU Rank':>10}"
    f"{'Ranks Match':>13}"
)
print(header)
print("-" * len(header))

for name in mt_models:
    llm_rank = llm_ranked.index(name) + 1
    bleu_rank = bleu_ranked.index(name) + 1
    match = "Yes" if llm_rank == bleu_rank else "No"
    print(
        f"{name:<22}"
        f"{judge_averages[name]['overall']:>13.2f}"
        f"{llm_rank:>10}"
        f"{bleu_scores[name]:>8.2f}"
        f"{bleu_rank:>10}"
        f"{match:>13}"
    )

agreement_count = sum(
    1 for n in mt_models
    if (llm_ranked.index(n) + 1) == (bleu_ranked.index(n) + 1)
)
print(f"\nModels with matching ranks: {agreement_count}/{len(mt_models)}")


# ---------------------------------------------------------------------------
# Print: per-sentence judge comments — best vs worst LLM-ranked model
# ---------------------------------------------------------------------------

best_model = llm_ranked[0]
worst_model = llm_ranked[-1]

print(f"\n{'=' * 70}")
print(f"JUDGE COMMENTS — {best_model} (ranked 1st) vs {worst_model} (ranked last)")
print("=" * 70)

for i, label in enumerate(LABELS):
    print(f"\n[{label}]")
    print(f"  Source:  {SOURCES[i]}")
    best_s = judge_scores[best_model][i]
    worst_s = judge_scores[worst_model][i]
    print(f"  {best_model:<20} (overall {best_s['overall']:.1f}): {best_s['comment']}")
    print(f"  {worst_model:<20} (overall {worst_s['overall']:.1f}): {worst_s['comment']}")


# ---------------------------------------------------------------------------
# Export full results to JSON
# ---------------------------------------------------------------------------

output = {
    "judge_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "models": mt_models,
    "sources": SOURCES,
    "references": REFERENCES,
    "translations": all_translations,
    "judge_scores": {
        name: {str(i): scores for i, scores in sent_scores.items()}
        for name, sent_scores in judge_scores.items()
    },
    "judge_averages": judge_averages,
    "bleu_scores": bleu_scores,
    "llm_ranking": llm_ranked,
    "bleu_ranking": bleu_ranked,
}

out_path = os.path.join(os.path.dirname(__file__), "judge_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nFull results saved to {out_path}")
