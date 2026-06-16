import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

# NLLB-200 (No Language Left Behind) is Meta's 2022 successor to mBART.
# Key differences vs mBART-50:
#   - Covers 200 languages vs 50
#   - Uses FLORES-200 language codes (e.g. "eng_Latn") instead of mBART's "en_XX"
#   - The 600M distilled variant is accurate enough for comparison and runs on modest hardware
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = NllbTokenizerFast.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def translate(text, model, tokenizer, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # NLLB requires forced_bos_token_id to specify the target language, same principle as mBART-50
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        output = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)

    translated_txt = tokenizer.decode(output[0], skip_special_tokens=True)
    return translated_txt


# NLLB uses FLORES-200 language codes: {language}_{script}
src_lang = "eng_Latn"
tgt_lang = "deu_Latn"

sample_txt = ['I cannot find any example code for XLM-E.', 'Hello world test one']
translated_txts = [translate(text, model, tokenizer, src_lang, tgt_lang) for text in sample_txt]

print("Original Texts: ", sample_txt)
print("Translated Texts: ", translated_txts)
