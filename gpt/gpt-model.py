import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# model_name = "EleutherAI/gpt-neo-2.7B"
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 has no dedicated pad token; reuse eos so padding doesn't error
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id


def translate(text, model, tokenizer, src_lang, tgt_lang):
    prompt = f"Translate the following text from {src_lang} to {tgt_lang}: {text}"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    translated_txt = tokenizer.decode(output[0], skip_special_tokens=True)

    return translated_txt


src_lang = "English"
tgt_lang = "German"

sample_txt = ['I cannot find any example code for XLM-E.', 'Hello world test one']
translated_txts = [translate(text, model, tokenizer, src_lang, tgt_lang) for text in sample_txt]

print("Original Texts: ", sample_txt)
print("Translated Texts: ", translated_txts)
