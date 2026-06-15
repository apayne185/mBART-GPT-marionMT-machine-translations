import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model_name = 'facebook/mbart-large-50-many-to-many-mmt'      # supports 50 langs
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)


def translate(text, model, tokenizer, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # forced_bos_token_id tells the decoder which language to generate — required for mBART-50
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    with torch.no_grad():
        output = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
    translated_txt = tokenizer.decode(output[0], skip_special_tokens=True)

    return translated_txt


src_lang = 'en_XX'
tgt_lang = 'de_DE'

sample_txt = ['I cannot find any example code for XLM-E.', 'Hello world test one']
translated_txts = [translate(text, model, tokenizer, src_lang, tgt_lang) for text in sample_txt]

print("Original Texts: ", sample_txt)
print("Translated Texts: ", translated_txts)


