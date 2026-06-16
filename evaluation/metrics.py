import warnings
import numpy as np
import nltk
from sacrebleu.metrics import BLEU, CHRF
from bert_score import score as _bert_score

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.translate.meteor_score import single_meteor_score

_labse_model = None


def _get_labse():
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        _labse_model = SentenceTransformer("LaBSE")
    return _labse_model


def compute_labse(texts_a, texts_b):
    """
    Cross-lingual cosine similarity via LaBSE (averaged over the corpus).
    Called as compute_labse(sources, hypotheses) to measure meaning preservation
    from source to translation — no reference translation needed.
    """
    model = _get_labse()
    emb_a = model.encode(list(texts_a), normalize_embeddings=True)
    emb_b = model.encode(list(texts_b), normalize_embeddings=True)
    scores = [float(np.dot(a, b)) for a, b in zip(emb_a, emb_b)]
    return round(sum(scores) / len(scores) * 100, 2)


def compute_bleu(hypotheses, references):
    # effective_order=True gives meaningful scores even for short sentences
    return round(BLEU(effective_order=True).corpus_score(hypotheses, [references]).score, 2)


def compute_chrf(hypotheses, references):
    # chrF measures character n-gram overlap — more robust than BLEU for morphologically rich languages
    return round(CHRF().corpus_score(hypotheses, [references]).score, 2)


def compute_meteor(hypotheses, references):
    scores = [
        single_meteor_score(ref.split(), hyp.split())
        for hyp, ref in zip(hypotheses, references)
    ]
    return round(sum(scores) / len(scores) * 100, 2)


def compute_bert_score(hypotheses, references, lang="de"):
    # BERTScore uses contextual embeddings to measure semantic similarity, not surface overlap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f1 = _bert_score(hypotheses, references, lang=lang, verbose=False)
    return round(f1.mean().item() * 100, 2)


def evaluate(hypotheses, references, lang="de"):
    """
    Compute reference-based MT metrics for a list of hypotheses vs references.

    Returns a dict with BLEU, chrF, METEOR, and BERTScore F1 (all scaled 0-100).
    For the reference-free LaBSE cross-lingual score, call compute_labse() separately.
    """
    return {
        "BLEU":          compute_bleu(hypotheses, references),
        "chrF":          compute_chrf(hypotheses, references),
        "METEOR":        compute_meteor(hypotheses, references),
        "BERTScore F1":  compute_bert_score(hypotheses, references, lang=lang),
    }
