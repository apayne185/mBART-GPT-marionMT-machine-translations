"""
Shared model loading functions used by all evaluation pipelines.

Each loader accepts tgt_lang (default "de") and returns (translate_fn, objects_to_free):
  - translate_fn(text: str) -> str
  - objects_to_free: list of objects to delete after inference to reclaim memory

Supported target languages: de (German), es (Spanish), ar (Arabic).
Language-specific codes are resolved via lang_config.LANG_CONFIG.

MarianMT: downloads a separate model per language pair (Helsinki-NLP/opus-mt-en-{tgt}).
mBART-50 / NLLB-200: multilingual — same model, different language code per call.
GPT-2 / TowerInstruct: prompted causal LMs — language name is injected into the prompt.
"""

import torch
from lang_config import LANG_CONFIG, MBART_SRC, NLLB_SRC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_marianmt(tgt_lang: str = "de"):
    cfg = LANG_CONFIG[tgt_lang]
    from transformers import MarianMTModel, MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained(cfg["marian_id"])
    model = MarianMTModel.from_pretrained(cfg["marian_id"]).to(device)

    def translate(text):
        inputs = {k: v.to(device) for k, v in
                  tokenizer(text, return_tensors="pt", padding=True, truncation=True).items()}
        with torch.no_grad():
            output = model.generate(**inputs, num_beams=4, max_new_tokens=256)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_mbart(tgt_lang: str = "de"):
    cfg = LANG_CONFIG[tgt_lang]
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    ).to(device)
    forced_bos = tokenizer.lang_code_to_id[cfg["mbart_tgt"]]

    def translate(text):
        tokenizer.src_lang = MBART_SRC
        inputs = {k: v.to(device) for k, v in
                  tokenizer(text, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).items()}
        with torch.no_grad():
            output = model.generate(
                **inputs, forced_bos_token_id=forced_bos, num_beams=4, max_new_tokens=256
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_nllb(tgt_lang: str = "de"):
    cfg = LANG_CONFIG[tgt_lang]
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
    tokenizer = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    ).to(device)
    forced_bos = tokenizer.convert_tokens_to_ids(cfg["nllb_tgt"])

    def translate(text):
        tokenizer.src_lang = NLLB_SRC
        inputs = {k: v.to(device) for k, v in
                  tokenizer(text, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).items()}
        with torch.no_grad():
            output = model.generate(
                **inputs, forced_bos_token_id=forced_bos, num_beams=4, max_new_tokens=256
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return translate, [model, tokenizer]


def load_gpt2(tgt_lang: str = "de"):
    cfg = LANG_CONFIG[tgt_lang]
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.config.pad_token_id = tokenizer.eos_token_id
    lang_name = cfg["name"]

    def translate(text):
        prompt = f"Translate the following text from English to {lang_name}: {text}"
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


def load_towerinstruct(tgt_lang: str = "de"):
    cfg = LANG_CONFIG[tgt_lang]
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
    lang_name = cfg["name"]

    def translate(text):
        prompt = (
            f"<s>[INST] Translate the following text from English into {lang_name}.\n"
            f"English: {text}\n{lang_name}: [/INST]"
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


def build_registry(tgt_lang: str = "de") -> dict:
    """
    Return the ordered dict of model name → loader function for en→tgt_lang.

    Supported: de (German), es (Spanish), ar (Arabic).
    TowerInstruct-7B is included only when a CUDA GPU is available.
    """
    if tgt_lang not in LANG_CONFIG:
        raise ValueError(
            f"Unsupported language {tgt_lang!r}. Choose from: {list(LANG_CONFIG)}"
        )
    registry = {
        "MarianMT": lambda: load_marianmt(tgt_lang),
        "mBART-50": lambda: load_mbart(tgt_lang),
        "NLLB-200": lambda: load_nllb(tgt_lang),
        "GPT-2":    lambda: load_gpt2(tgt_lang),
    }
    if torch.cuda.is_available():
        registry["TowerInstruct-7B"] = lambda: load_towerinstruct(tgt_lang)
    else:
        print("No CUDA detected — TowerInstruct-7B will be skipped.\n")
    return registry
