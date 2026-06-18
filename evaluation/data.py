# Shared evaluation sentences used by evaluation/run_comparison.py,
# semantic_analysis/semantic_similarity.py, and langchain_pipeline/pipeline.py.
# Edit SOURCES and LABELS here to update all pipelines at once.
#
# Reference translations: idiomatic where applicable (not literal).
# Sentence 4 ("raining cats and dogs") uses the natural idiom in each language
# rather than a word-for-word rendering — this tests whether MT models
# preserve meaning or blindly translate surface form.
#
# Arabic references verified against standard Modern Standard Arabic (MSA).
# Spanish references use Castilian conventions.

SOURCES = [
    "The weather is beautiful today.",
    "Machine translation systems have improved significantly over the past decade.",
    "Neural networks learn representations from data using gradient descent.",
    "It's raining cats and dogs.",
    "I cannot find any example code for XLM-E.",
    "The conference on natural language processing attracted researchers from around the world.",
]

REFERENCES = {
    "de": [
        "Das Wetter ist heute wunderbar.",
        "Maschinelle Übersetzungssysteme haben sich im vergangenen Jahrzehnt erheblich verbessert.",
        "Neuronale Netze lernen Repräsentationen aus Daten mithilfe des Gradientenabstiegs.",
        "Es regnet in Strömen.",
        "Ich kann keinen Beispielcode für XLM-E finden.",
        "Die Konferenz über natürliche Sprachverarbeitung zog Forscher aus aller Welt an.",
    ],
    "es": [
        "El tiempo está hermoso hoy.",
        "Los sistemas de traducción automática han mejorado significativamente en la última década.",
        "Las redes neuronales aprenden representaciones de los datos mediante el descenso del gradiente.",
        "Está lloviendo a cántaros.",
        "No puedo encontrar ningún código de ejemplo para XLM-E.",
        "La conferencia sobre procesamiento del lenguaje natural atrajo a investigadores de todo el mundo.",
    ],
    "ar": [
        "الطقس جميل اليوم.",
        "تحسنت أنظمة الترجمة الآلية بشكل ملحوظ خلال العقد الماضي.",
        "تتعلم الشبكات العصبية تمثيلات من البيانات باستخدام الانحدار التدريجي.",
        "تهطل الأمطار بغزارة.",
        "لا أستطيع إيجاد أي كود نموذجي لـ XLM-E.",
        "استقطب المؤتمر المعني بمعالجة اللغة الطبيعية باحثين من جميع أنحاء العالم.",
    ],
}

# Short labels for visualisation axes — must stay aligned with SOURCES
LABELS = [
    "Weather",
    "MT systems",
    "Neural nets",
    "Idiom (rain)",
    "XLM-E code",
    "NLP conference",
]


def get_references(tgt_lang: str = "de") -> list:
    """Return the reference translations for the given target language."""
    if tgt_lang not in REFERENCES:
        raise ValueError(
            f"No references for {tgt_lang!r}. Available: {list(REFERENCES)}"
        )
    return REFERENCES[tgt_lang]
