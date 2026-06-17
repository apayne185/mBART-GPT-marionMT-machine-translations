"""
Language configuration for all-pairs multilingual evaluation.

Each entry maps a two-letter ISO language code to the identifiers needed
by every model family and dataset in this project.

Supported: en (English), de (German), es (Spanish), ar (Arabic)

To add a new language:
  1. Add an entry here with the correct codes for each model family.
  2. Add reference translations to data.py (for run_comparison.py — en→X only).
  3. If OPUS-100 support is needed (en source only), add opus_pair and opus_key.

Field reference:
  marian_code  Two-letter code used in Helsinki-NLP/opus-mt-{src}-{tgt} model names.
               The full model ID is constructed dynamically from src and tgt codes.
               Not all src→tgt pairs have a direct model; missing ones are skipped.
  mbart_code   mBART-50 language token — used as src_lang or forced_bos_token.
  nllb_code    FLORES-200 language code — used by NLLB-200 and for FLORES-200 corpus loading.
  bert_lang    Language code for BERTScore (typically the ISO code of the target language).
  opus_pair    OPUS-100 config name (alphabetical pair order, English-centric pairs only).
  opus_key     Key for the target-language side in OPUS-100 "translation" dicts.

mBART-50 language codes: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
NLLB-200 / FLORES-200 codes: https://github.com/facebookresearch/flores
MarianMT models: https://huggingface.co/Helsinki-NLP
"""

LANG_CONFIG = {
    "en": {
        "name":        "English",
        "marian_code": "en",
        "mbart_code":  "en_XX",
        "nllb_code":   "eng_Latn",
        "bert_lang":   "en",
    },
    "de": {
        "name":        "German",
        "marian_code": "de",
        "mbart_code":  "de_DE",
        "nllb_code":   "deu_Latn",
        "bert_lang":   "de",
        # OPUS-100 config name (alphabetical pair order) and target key
        # Used by load_opus100_pairs — English-source evaluation only
        "opus_pair":   "de-en",
        "opus_key":    "de",
    },
    "es": {
        "name":        "Spanish",
        "marian_code": "es",
        "mbart_code":  "es_XX",
        "nllb_code":   "spa_Latn",
        "bert_lang":   "es",
        "opus_pair":   "en-es",
        "opus_key":    "es",
    },
    "ar": {
        "name":        "Arabic",
        "marian_code": "ar",
        "mbart_code":  "ar_AR",
        "nllb_code":   "arb_Arab",
        "bert_lang":   "ar",
        "opus_pair":   "ar-en",
        "opus_key":    "ar",
    },
}
