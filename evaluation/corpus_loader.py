"""
Corpus loaders for multilingual MT evaluation.

load_opus100_pairs   — OPUS-100 (English-centric, en→X only, up to 2000 test pairs)
load_flores200_pairs — FLORES-200 (multi-way parallel, any src→tgt, 1012 devtest sentences)

FLORES-200 is the preferred corpus for non-English-source pairs and for cross-language
comparisons because all 1012 sentences are professionally translated and index-aligned
across all 200 languages. This makes BLEU scores directly comparable across any
language direction (e.g., en→de vs de→en vs es→ar).

Dataset references:
  OPUS-100:   Helsinki-NLP/opus-100 on HuggingFace
  FLORES-200: facebook/flores on HuggingFace (configs named by NLLB/FLORES-200 codes)
"""

import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from lang_config import LANG_CONFIG


def load_opus100_pairs(tgt_lang: str = "de", n: int = 100, split: str = "test"):
    """Load en→tgt_lang pairs from OPUS-100.

    OPUS-100 is English-centric — all configs pair English with one other language.
    Config naming follows alphabetical order (e.g., "de-en", "en-es", "ar-en").
    A fallback tries the reverse ordering in case of inconsistency.

    Only usable for English-source evaluation. Use load_flores200_pairs for all others.
    """
    cfg = LANG_CONFIG[tgt_lang]
    key = cfg["opus_key"]
    primary = cfg["opus_pair"]
    reverse = "-".join(reversed(primary.split("-")))
    for pair in [primary, reverse]:
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", pair, split=split)
            subset = ds.select(range(min(n, len(ds))))
            sources    = [row["translation"]["en"] for row in subset]
            references = [row["translation"][key]  for row in subset]
            return sources, references
        except Exception:
            continue
    raise ValueError(
        f"Could not load OPUS-100 for '{tgt_lang}'. Tried: {primary!r}, {reverse!r}. "
        f"Check configs at huggingface.co/datasets/Helsinki-NLP/opus-100."
    )


def load_flores200_pairs(src_lang: str, tgt_lang: str, n: int = 100, split: str = "devtest"):
    """Load src→tgt sentence pairs for MT evaluation.

    Tries sources in order:
      1. facebook/flores  — official FLORES-200, gated (requires HuggingFace access)
      2. gsarti/flores_101 — public FLORES-101, same 1012 sentences, needs datasets<3.0
      3. OPUS-100          — fallback for English-involving pairs (en↔X both directions)

    Cross-lingual pairs (neither language English) require FLORES. Request access at
    https://huggingface.co/datasets/facebook/flores — usually approved within minutes.
    """
    # Try FLORES (official or community mirror)
    for flores_name in ("facebook/flores", "gsarti/flores_101"):
        try:
            src_code = LANG_CONFIG[src_lang]["flores_code" if "101" in flores_name else "nllb_code"]
            tgt_code = LANG_CONFIG[tgt_lang]["flores_code" if "101" in flores_name else "nllb_code"]
            src_ds = load_dataset(flores_name, src_code, split=split)
            tgt_ds = load_dataset(flores_name, tgt_code, split=split)
            n = min(n, len(src_ds), len(tgt_ds))
            return [src_ds[i]["sentence"] for i in range(n)], \
                   [tgt_ds[i]["sentence"] for i in range(n)]
        except Exception:
            continue

    # OPUS-100 fallback for English-involving pairs
    if src_lang == "en":
        return load_opus100_pairs(tgt_lang=tgt_lang, n=n, split="test")
    if tgt_lang == "en":
        return _load_opus100_reversed(src_lang=src_lang, n=n)

    raise ValueError(
        f"No corpus available for {src_lang}→{tgt_lang}. "
        f"Cross-lingual pairs require FLORES-200 access.\n"
        f"Request it at: https://huggingface.co/datasets/facebook/flores"
    )


def _load_opus100_reversed(src_lang: str, n: int = 100, split: str = "test"):
    """Load X→en pairs from OPUS-100 by treating the non-English side as source."""
    cfg = LANG_CONFIG[src_lang]
    key = cfg["opus_key"]
    primary = cfg["opus_pair"]
    reverse = "-".join(reversed(primary.split("-")))
    for pair in [primary, reverse]:
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", pair, split=split)
            subset = ds.select(range(min(n, len(ds))))
            sources    = [row["translation"][key]  for row in subset]
            references = [row["translation"]["en"] for row in subset]
            return sources, references
        except Exception:
            continue
    raise ValueError(f"Could not load OPUS-100 for {src_lang}→en.")
