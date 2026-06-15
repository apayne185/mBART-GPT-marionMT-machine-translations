import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# TowerInstruct is a LLaMA-2-based model fine-tuned specifically for translation tasks.
# Unlike mBART/MarianMT (encoder-decoder), this is a decoder-only LLM guided by instruction prompting.
model_name = "Unbabel/TowerInstruct-7B-v0.2"

# 4-bit quantization reduces memory from ~14GB to ~4-5GB with minimal quality loss
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)


def translate(text, model, tokenizer, src_lang, tgt_lang):
    # TowerInstruct uses the LLaMA-2 chat format; the target language token is left open for generation
    prompt = (
        f"<s>[INST] Translate the following text from {src_lang} into {tgt_lang}.\n"
        f"{src_lang}: {text}\n"
        f"{tgt_lang}: [/INST]"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens, not the prompt
    generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return generated.strip()


src_lang = "English"
tgt_lang = "German"

sample_txt = ['I cannot find any example code for XLM-E.', 'Hello world test one']
translated_txts = [translate(text, model, tokenizer, src_lang, tgt_lang) for text in sample_txt]

print("Original Texts: ", sample_txt)
print("Translated Texts: ", translated_txts)
