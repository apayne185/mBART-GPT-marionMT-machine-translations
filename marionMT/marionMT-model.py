import torch
from transformers import MarianMTModel, MarianTokenizer

# MarianMT is a lightweight encoder-decoder model trained on language-pair-specific corpora.
# For cross-lingual semantic similarity (measuring meaning preservation across languages),
# see labse/labse-model.py which uses a shared multilingual embedding space instead.

src_lang = 'en'   # source language
tgt_lang = 'de'   # target language

# model_name = "xlm-roberta-base"
model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs)
    translated_txt = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_txt


sample_txt = ['I cannot find any example code for XLM-E.', 'Hello world test one']
translated_txts = [translate(text, model, tokenizer) for text in sample_txt]

print("Original Texts: ", sample_txt)
print("Translated Texts: ", translated_txts)
