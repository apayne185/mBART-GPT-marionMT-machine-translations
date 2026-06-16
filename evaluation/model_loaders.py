"""
Shared model loading functions used by evaluation/run_comparison.py
and semantic_analysis/semantic_similarity.py.

Each loader returns (translate_fn, objects_to_free):
  - translate_fn(text: str) -> str
  - objects_to_free: list of objects to delete after inference to reclaim memory
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_marianmt():
    from transformers import MarianMTModel, MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(device)

    def translate(text):
        inputs = {k: v.to(device) for k, v in
                  tokenizer(text, return_tensors="pt", padding=True, truncation=True).items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_mbart():
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    ).to(device)
    forced_bos = tokenizer.lang_code_to_id["de_DE"]

    def translate(text):
        tokenizer.src_lang = "en_XX"
        inputs = {k: v.to(device) for k, v in
                  tokenizer(text, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).items()}
        with torch.no_grad():
            output = model.generate(**inputs, forced_bos_token_id=forced_bos, max_new_tokens=256)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_nllb():
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
    tokenizer = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    ).to(device)
    forced_bos = tokenizer.convert_tokens_to_ids("deu_Latn")

    def translate(text):
        tokenizer.src_lang = "eng_Latn"
        inputs = {k: v.to(device) for k, v in
                  tokenizer(text, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).items()}
        with torch.no_grad():
            output = model.generate(**inputs, forced_bos_token_id=forced_bos, max_new_tokens=256)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_gpt2():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.config.pad_token_id = tokenizer.eos_token_id

    def translate(text):
        prompt = f"Translate the following text from English to German: {text}"
        inputs = {k: v.to(device) for k, v in
                  tokenizer(prompt, return_tensors="pt", padding=True).items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_towerinstruct():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-7B-v0.2")
    model = AutoModelForCausalLM.from_pretrained(
        "Unbabel/TowerInstruct-7B-v0.2",
        quantization_config=quantization_config,
        device_map="auto",
    )

    def translate(text):
        prompt = (
            f"<s>[INST] Translate the following text from English into German.\n"
            f"English: {text}\nGerman: [/INST]"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()

    return translate, [model, tokenizer]


def build_registry():
    """Return the ordered dict of model name → loader function."""
    registry = {
        "MarianMT":  load_marianmt,
        "mBART-50":  load_mbart,
        "NLLB-200":  load_nllb,
        "GPT-2":     load_gpt2,
    }
    if torch.cuda.is_available():
        registry["TowerInstruct-7B"] = load_towerinstruct
    else:
        print("No CUDA detected — TowerInstruct-7B will be skipped.\n")
    return registry
