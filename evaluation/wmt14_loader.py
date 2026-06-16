"""
WMT14 newstest2014 loader (en → de).

WMT14 is the standard academic MT benchmark — used to evaluate the original
Transformer ("Attention is All You Need", Vaswani et al. 2017) and most
subsequent MT research. The test split (newstest2014) contains 3003 sentence
pairs from the news domain, professionally translated.

This replaces the 6 hand-crafted sentences in data.py for statistically
meaningful corpus-level evaluation.
"""

from datasets import load_dataset


def load_wmt14_pairs(n: int = 100, split: str = "test"):
    """
    Load the first n en→de sentence pairs from WMT14 newstest2014.

    Args:
        n:     Number of sentences to load (default 100; max 3003 for test split).
               100 gives statistically stable BLEU while keeping runtime under ~10 min
               on CPU. Use n=3003 for full-benchmark results.
        split: HuggingFace split name ("test" = newstest2014).

    Returns:
        (sources, references): two lists of strings, both length n.
    """
    ds = load_dataset("wmt/wmt14", "de-en", split=split)
    subset = ds.select(range(min(n, len(ds))))
    sources = [row["translation"]["en"] for row in subset]
    references = [row["translation"]["de"] for row in subset]
    return sources, references
