# Shared evaluation sentences used by both evaluation/run_comparison.py
# and semantic_analysis/semantic_similarity.py. Edit here to change both.

SOURCES = [
    "The weather is beautiful today.",
    "Machine translation systems have improved significantly over the past decade.",
    "Neural networks learn representations from data using gradient descent.",
    "It's raining cats and dogs.",
    "I cannot find any example code for XLM-E.",
    "The conference on natural language processing attracted researchers from around the world.",
]

# Human reference translations (en → de)
REFERENCES = [
    "Das Wetter ist heute wunderbar.",
    "Maschinelle Übersetzungssysteme haben sich im vergangenen Jahrzehnt erheblich verbessert.",
    "Neuronale Netze lernen Repräsentationen aus Daten mithilfe des Gradientenabstiegs.",
    "Es regnet in Strömen.",
    "Ich kann keinen Beispielcode für XLM-E finden.",
    "Die Konferenz über natürliche Sprachverarbeitung zog Forscher aus aller Welt an.",
]

# Short labels for visualisation axes — must stay aligned with SOURCES
LABELS = [
    "Weather",
    "MT systems",
    "Neural nets",
    "Idiom (rain)",
    "XLM-E code",
    "NLP conference",
]
