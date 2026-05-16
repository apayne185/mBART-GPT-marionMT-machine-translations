from transformers import MBartForConditionalGeneration, MBartTokenizer


# text_classification = pipeline("text-classification", model="EleutherAI/gpt-neo-2.7B")
# result = text_classification("This is an amazing day!")
# print(result)

model_name = 'facebook/mbart-large-50-many-to-many-mmt'      #supports 50 langs
tokenizer = MBartTokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)


def translate(text, model, tokenizer, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output = model.generate(**inputs)
    translated_txt = tokenizer.decode(output[0], skip_special_tokens=True)

    return translated_txt

src_lang = 'en_XX'   
tgt_lang = 'de_XX'


sample_txt = ['I cannot find any example code for XLM-E.', 'Hello world test one']
translated_txts = [translate(text,model,tokenizer,src_lang, tgt_lang) for text in sample_txt]

print("Original Texts: ", sample_txt)
print("Translated Texts: ", translated_txts)


