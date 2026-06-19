import warnings
import numpy as np
import nltk
from sacrebleu.metrics import BLEU, CHRF

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.translate.meteor_score import single_meteor_score
from bert_score import score as _bert_score

_labse_model = None
_comet_model = None


def _get_labse():
    global _labse_model
    if _labse_model is None:
        from sentence_transformers import SentenceTransformer
        _labse_model = SentenceTransformer("LaBSE")
    return _labse_model


def _get_comet():
    """Load COMET model once and cache it for the process lifetime.

    Downloads Unbabel/wmt22-comet-da on first call (~1.7 GB, cached locally).
    Returns None if unbabel-comet is not installed.
    """
    global _comet_model
    if _comet_model is None:
        try:
            from comet import download_model, load_from_checkpoint
            print("Loading COMET model (Unbabel/wmt22-comet-da)...")
            path = download_model("Unbabel/wmt22-comet-da")
            _comet_model = load_from_checkpoint(path)
        except ImportError:
            return None
    return _comet_model


def compute_labse(texts_a, texts_b):
    """Cross-lingual cosine similarity via LaBSE (averaged over corpus, scaled 0–100).

    Called as compute_labse(sources, hypotheses) — reference-free.
    """
    model = _get_labse()
    emb_a = model.encode(list(texts_a), normalize_embeddings=True)
    emb_b = model.encode(list(texts_b), normalize_embeddings=True)
    scores = [float(np.dot(a, b)) for a, b in zip(emb_a, emb_b)]
    return round(sum(scores) / len(scores) * 100, 2)


def compute_comet(sources, hypotheses, references):
    """COMET score via Unbabel/wmt22-comet-da (scaled 0–100).

    COMET is a learned metric trained on human translation quality judgements.
    It correlates more strongly with human evaluation than BLEU or chrF.

    Returns None if unbabel-comet is not installed (pip install unbabel-comet).
    First call downloads the model (~1.7 GB) and caches it locally.
    """
    model = _get_comet()
    if model is None:
        return None
    import torch
    data = [{"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)]
    gpus = 1 if torch.cuda.is_available() else 0
    result = model.predict(data, batch_size=8, gpus=gpus, progress_bar=False)
    return round(float(result.system_score) * 100, 2)


def compute_bleu(hypotheses, references):
    return round(BLEU(effective_order=True).corpus_score(hypotheses, [references]).score, 2)


def compute_chrf(hypotheses, references):
    return round(CHRF().corpus_score(hypotheses, [references]).score, 2)


def compute_meteor(hypotheses, references):
    scores = [
        single_meteor_score(ref.split(), hyp.split())
        for hyp, ref in zip(hypotheses, references)
    ]
    return round(sum(scores) / len(scores) * 100, 2)


def compute_bert_score(hypotheses, references, lang="de"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, f1 = _bert_score(hypotheses, references, lang=lang, verbose=False)
    return round(f1.mean().item() * 100, 2)


def evaluate(hypotheses, references, lang="de", sources=None):
    """Compute MT evaluation metrics for a list of hypotheses vs references.

    Returns a dict with BLEU, chrF, METEOR, BERTScore F1 (all scaled 0–100).
    If sources are provided and unbabel-comet is installed, COMET is also included.
    For the reference-free LaBSE cross-lingual score, call compute_labse() separately.
    """
    scores = {
        "BLEU":          compute_bleu(hypotheses, references),
        "chrF":          compute_chrf(hypotheses, references),
        "METEOR":        compute_meteor(hypotheses, references),
        "BERTScore F1":  compute_bert_score(hypotheses, references, lang=lang),
    }
    if sources is not None:
        comet = compute_comet(sources, hypotheses, references)
        if comet is not None:
            scores["COMET"] = comet
    return scores
