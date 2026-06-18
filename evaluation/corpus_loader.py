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
    """Load src→tgt sentence pairs from FLORES-101 (gsarti/flores_101).

    FLORES-101 is a multi-way parallel benchmark covering 101 languages. All
    1012 devtest sentences are professionally translated and index-aligned across
    every language config, making scores directly comparable across any direction.

    FLORES-200 (facebook/flores) uses the same source sentences but is gated on
    HuggingFace. gsarti/flores_101 is publicly accessible and covers all four
    supported languages (en, de, es, ar).

    Config names use ISO 639-3 codes (stored in lang_config under flores_code).
    Available splits: devtest (1012 sentences), dev (997 sentences).
    """
    src_code = LANG_CONFIG[src_lang]["flores_code"]
    tgt_code = LANG_CONFIG[tgt_lang]["flores_code"]

    src_ds = load_dataset("gsarti/flores_101", src_code, split=split)
    tgt_ds = load_dataset("gsarti/flores_101", tgt_code, split=split)

    n = min(n, len(src_ds), len(tgt_ds))
    sources    = [src_ds[i]["sentence"] for i in range(n)]
    references = [tgt_ds[i]["sentence"] for i in range(n)]
    return sources, references
