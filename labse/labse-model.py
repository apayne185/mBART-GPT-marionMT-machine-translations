from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# LaBSE (Language-agnostic BERT Sentence Embeddings) is not a translation model.
# It maps text from 109 languages into a shared embedding space, so semantically
# equivalent sentences across languages land close together regardless of surface form.
#
# Role in this project: semantic evaluation layer.
# Given a source sentence and its translation, LaBSE measures how much meaning
# was preserved — something BLEU cannot capture since it only checks word overlap.
model = SentenceTransformer("LaBSE")


def embed(texts):
    return model.encode(texts, normalize_embeddings=True)


def semantic_similarity(source_texts, translated_texts):
    src_embeddings = embed(source_texts)
    tgt_embeddings = embed(translated_texts)

    scores = [
        float(cosine_similarity([src], [tgt])[0][0])
        for src, tgt in zip(src_embeddings, tgt_embeddings)
    ]
    return scores


# Example: compare English source against a German translation.
# High cosine similarity (close to 1.0) means the meaning was well preserved.
source_texts = [
    "I cannot find any example code for XLM-E.",
    "Hello world test one",
]

# Reference translations (human or model-generated) to score against
translated_texts = [
    "Ich kann keinen Beispielcode für XLM-E finden.",
    "Hallo Welt Test eins",
]

scores = semantic_similarity(source_texts, translated_texts)

for src, tgt, score in zip(source_texts, translated_texts, scores):
    print(f"Source:      {src}")
    print(f"Translation: {tgt}")
    print(f"Similarity:  {score:.4f}")
    print()
