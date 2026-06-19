"""
Shared model loading functions used by all evaluation pipelines.

Each loader accepts src_lang and tgt_lang and returns (translate_fn, objects_to_free):
  - translate_fn(texts: list[str]) -> list[str]  — batched inference
  - objects_to_free: list of objects to delete after inference to reclaim memory

Inputs are processed in chunks of BATCH_SIZE to keep peak memory predictable.
On CPU, BATCH_SIZE=8 gives a 2-4× speedup over one-at-a-time inference for
seq2seq models (MarianMT, mBART-50, NLLB-200). On GPU the gain is 8-16×.

Causal LMs (GPT-2, TowerInstruct) use left-padding so that all generated tokens
in a batch start at the same offset (inputs["input_ids"].shape[1]).

Supported languages: en (English), de (German), es (Spanish), ar (Arabic).
Language-specific codes are resolved via lang_config.LANG_CONFIG.

MarianMT: constructs Helsinki-NLP/opus-mt-{src}-{tgt} from marian_code fields.
          Not all src→tgt pairs have a direct model; missing ones raise an error
          which run_multilang.py catches and skips gracefully.
mBART-50 / NLLB-200: multilingual — same weights, different language codes per call.
GPT-2 / TowerInstruct: English-source only (prompted causal LMs).
"""

import torch
from lang_config import LANG_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8  # sentences per forward pass; lower if you hit OOM on CPU


def load_marianmt(src_lang: str = "en", tgt_lang: str = "de"):
    src_cfg = LANG_CONFIG[src_lang]
    tgt_cfg = LANG_CONFIG[tgt_lang]
    model_id = f"Helsinki-NLP/opus-mt-{src_cfg['marian_code']}-{tgt_cfg['marian_code']}"

    from transformers import MarianMTModel, MarianTokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id).to(device)
    model.generation_config.max_length = None  # avoid conflict with max_new_tokens

    def translate(texts: list[str]) -> list[str]:
        results = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = {k: v.to(device) for k, v in
                      tokenizer(batch, return_tensors="pt", padding=True, truncation=True).items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, num_beams=4, max_new_tokens=256)
            results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return results

    return translate, [model, tokenizer]


def load_mbart(src_lang: str = "en", tgt_lang: str = "de"):
    src_cfg = LANG_CONFIG[src_lang]
    tgt_cfg = LANG_CONFIG[tgt_lang]
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = src_cfg["mbart_code"]  # fixed for this loader instance
    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    ).to(device)
    model.generation_config.max_length = None  # avoid conflict with max_new_tokens
    forced_bos = tokenizer.lang_code_to_id[tgt_cfg["mbart_code"]]

    def translate(texts: list[str]) -> list[str]:
        results = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = {k: v.to(device) for k, v in
                      tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, forced_bos_token_id=forced_bos, num_beams=4, max_new_tokens=256
                )
            results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return results

    return translate, [model, tokenizer]


def load_nllb(src_lang: str = "en", tgt_lang: str = "de"):
    src_cfg = LANG_CONFIG[src_lang]
    tgt_cfg = LANG_CONFIG[tgt_lang]
    from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
    tokenizer = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer.src_lang = src_cfg["nllb_code"]  # fixed for this loader instance
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    ).to(device)
    model.generation_config.max_length = None  # avoid conflict with max_new_tokens
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_cfg["nllb_code"])

    def translate(texts: list[str]) -> list[str]:
        results = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = {k: v.to(device) for k, v in
                      tokenizer(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, forced_bos_token_id=forced_bos, num_beams=4, max_new_tokens=256
                )
            results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return results

    return translate, [model, tokenizer]


def load_gpt2(tgt_lang: str = "de"):
    cfg = LANG_CONFIG[tgt_lang]
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad so all new tokens start at the same offset
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.config.pad_token_id = tokenizer.eos_token_id
    lang_name = cfg["name"]

    def translate(texts: list[str]) -> list[str]:
        results = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            prompts = [
                f"Translate the following text from English to {lang_name}: {t}"
                for t in batch
            ]
            inputs = {k: v.to(device) for k, v in
                      tokenizer(prompts, return_tensors="pt", padding=True).items()}
            # With left-padding all sequences share the same padded length, so
            # new tokens are uniformly at positions [prompt_len:] for every sample.
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=256,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for out in outputs:
                results.append(tokenizer.decode(out[prompt_len:], skip_special_tokens=True))
        return results

    return translate, [model, tokenizer]


def load_towerinstruct(tgt_lang: str = "de"):
    cfg = LANG_CONFIG[tgt_lang]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-7B-v0.2")
    tokenizer.padding_side = "left"  # left-pad for batched causal generation
    model = AutoModelForCausalLM.from_pretrained(
        "Unbabel/TowerInstruct-7B-v0.2",
        quantization_config=quantization_config,
        device_map="auto",
    )
    lang_name = cfg["name"]

    def translate(texts: list[str]) -> list[str]:
        results = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            prompts = [
                f"<s>[INST] Translate the following text from English into {lang_name}.\n"
                f"English: {t}\n{lang_name}: [/INST]"
                for t in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=256, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            for out in outputs:
                results.append(tokenizer.decode(out[prompt_len:], skip_special_tokens=True).strip())
        return results

    return translate, [model, tokenizer]


def build_registry(src_lang: str = "en", tgt_lang: str = "de") -> dict:
    """
    Return the ordered dict of model name → loader function for src_lang→tgt_lang.

    GPT-2 and TowerInstruct are English-source only and are excluded when src_lang != "en".
    MarianMT is included for all pairs; if the direct model does not exist on HuggingFace
    the loader raises an error which callers can catch and skip.
    """
    for lang in (src_lang, tgt_lang):
        if lang not in LANG_CONFIG:
            raise ValueError(
                f"Unsupported language {lang!r}. Choose from: {list(LANG_CONFIG)}"
            )
    if src_lang == tgt_lang:
        raise ValueError(f"Source and target must differ (got {src_lang!r} for both).")

    registry = {
        "MarianMT": lambda: load_marianmt(src_lang, tgt_lang),
        "mBART-50": lambda: load_mbart(src_lang, tgt_lang),
        "NLLB-200": lambda: load_nllb(src_lang, tgt_lang),
    }

    if src_lang == "en":
        registry["GPT-2"] = lambda: load_gpt2(tgt_lang)
        if torch.cuda.is_available():
            registry["TowerInstruct-7B"] = lambda: load_towerinstruct(tgt_lang)
        else:
            print("No CUDA detected — TowerInstruct-7B will be skipped.\n")
    else:
        print(f"GPT-2 and TowerInstruct are English-source only — skipped for {src_lang}→{tgt_lang}.\n")

    return registry
