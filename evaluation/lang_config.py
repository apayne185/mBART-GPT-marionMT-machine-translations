"""
Language configuration for multi-language en→X evaluation.

Each entry maps a two-letter target language code to the model-specific
identifiers needed by every loader in model_loaders.py.

Supported targets: de (German), es (Spanish), ar (Arabic)

To add a new language:
  1. Add an entry here with the correct codes for each model family.
  2. Add reference translations to data.py.
  3. Verify the OPUS-100 config name at huggingface.co/datasets/opus100.

mBART-50 language codes: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
NLLB-200 language codes: FLORES-200 format — https://github.com/facebookresearch/flores
MarianMT models: https://huggingface.co/Helsinki-NLP
"""

MBART_SRC = "en_XX"
NLLB_SRC  = "eng_Latn"

LANG_CONFIG = {
    "de": {
        "name":       "German",
        "marian_id":  "Helsinki-NLP/opus-mt-en-de",
        "mbart_tgt":  "de_DE",
        "nllb_tgt":   "deu_Latn",
        "bert_lang":  "de",
        # OPUS-100 config name (alphabetical pair order) and target key
        "opus_pair":  "de-en",
        "opus_key":   "de",
    },
    "es": {
        "name":       "Spanish",
        "marian_id":  "Helsinki-NLP/opus-mt-en-es",
        "mbart_tgt":  "es_XX",
        "nllb_tgt":   "spa_Latn",
        "bert_lang":  "es",
        "opus_pair":  "en-es",
        "opus_key":   "es",
    },
    "ar": {
        "name":       "Arabic",
        "marian_id":  "Helsinki-NLP/opus-mt-en-ar",
        "mbart_tgt":  "ar_AR",
        "nllb_tgt":   "arb_Arab",
        "bert_lang":  "ar",
        "opus_pair":  "ar-en",
        "opus_key":   "ar",
    },
}
