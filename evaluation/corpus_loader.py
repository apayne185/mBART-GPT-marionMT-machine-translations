"""
OPUS-100 corpus loader for multi-language en→X evaluation.

OPUS-100 is a balanced, 100-language parallel corpus built from OPUS data.
Each language pair has up to 1M training sentences and 2000 test sentences,
professionally aligned from existing parallel corpora (EuroParl, CCAligned, etc.).

Using OPUS-100 (rather than WMT14) for the multi-language benchmark ensures
the same dataset format and test split size across all language pairs, making
cross-language score comparisons fair.

Note on pair naming: OPUS-100 uses alphabetical config names, so Arabic-English
is "ar-en" (not "en-ar") and German-English is "de-en". The translation dict
always contains both language codes as keys.

Usage:
    from corpus_loader import load_opus100_pairs
    sources, references = load_opus100_pairs("es", n=100)
    sources, references = load_opus100_pairs("ar", n=200)
"""

from datasets import load_dataset
from lang_config import LANG_CONFIG


def load_opus100_pairs(tgt_lang: str = "de", n: int = 100, split: str = "test"):
    """
    Load the first n en→tgt_lang sentence pairs from OPUS-100.

    Args:
        tgt_lang: Two-letter target language code ("de", "es", "ar").
                  Must be a key in LANG_CONFIG.
        n:        Number of sentence pairs to load (default 100, max 2000 for test).
        split:    Dataset split — "test" (2000 pairs), "validation" (2000 pairs),
                  or "train" (up to 1M pairs).

    Returns:
        (sources, references): two lists of strings, both length n.
        sources    — English sentences
        references — Target-language reference translations
    """
    if tgt_lang not in LANG_CONFIG:
        raise ValueError(
            f"Unsupported language {tgt_lang!r}. "
            f"Choose from: {list(LANG_CONFIG)}"
        )

    cfg = LANG_CONFIG[tgt_lang]
    pair = cfg["opus_pair"]
    key  = cfg["opus_key"]

    ds = load_dataset("opus100", pair, split=split)
    subset = ds.select(range(min(n, len(ds))))
    sources    = [row["translation"]["en"] for row in subset]
    references = [row["translation"][key]  for row in subset]
    return sources, references
