"""
LLM-as-Judge evaluation chain built with LangChain LCEL.

Composes four LangChain primitives using the pipe operator:
    ChatPromptTemplate | ChatHuggingFace | StrOutputParser | RunnableLambda(extract_json)

The local model (Qwen2.5-1.5B-Instruct by default) scores each machine
translation on three linguistic dimensions:
  - fluency:   grammatical correctness and naturalness in German
  - adequacy:  preservation of source meaning without omissions or distortions
  - style:     appropriateness of register and word choice

The JSON extraction step is a RunnableLambda in the chain, not a post-processing
hack — it keeps the pipeline composable and the output uniform regardless of
whether the model wraps its JSON in markdown fences or adds explanatory text.

Usage:
    chain, resources = build_judge_chain()
    result = chain.invoke({"source": "...", "translation": "..."})
    # result = {"fluency": 8, "adequacy": 9, "style": 7, "overall": 8.0, "comment": "..."}

    # Batch evaluation (sequential for local models):
    results = chain.batch(inputs)

    # Free GPU memory when done:
    resources.clear()
    gc.collect()
    torch.cuda.empty_cache()
"""

import json
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline as hf_pipeline,
)

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_SYSTEM = (
    "You are an expert German linguist and machine translation evaluator. "
    "You will be given an English source sentence and a German translation. "
    "Evaluate strictly and return only a valid JSON object — no markdown, no explanation."
)

_HUMAN = """\
Source (English): {source}
Translation (German): {translation}

Score this translation on three dimensions, each 1–10:
- fluency:   Is the German grammatically correct and natural-sounding?
- adequacy:  Does the translation convey the full meaning of the source without omissions or distortions?
- style:     Is the word choice idiomatic and appropriate for a native speaker?

Set "overall" to the arithmetic mean of the three scores (one decimal place).
Add a one-sentence "comment" on the most notable strength or weakness.

Return exactly this JSON and nothing else:
{{"fluency": <int>, "adequacy": <int>, "style": <int>, "overall": <float>, "comment": "<string>"}}"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])

_FALLBACK = {"fluency": 0, "adequacy": 0, "style": 0, "overall": 0.0, "comment": "parse error"}


def _extract_json(text: str) -> dict:
    """
    Extract a JSON object from model output.

    Local models sometimes wrap their response in markdown fences or add
    introductory text. This function finds the first well-formed JSON object
    by walking from the opening brace to its matching closing brace.
    """
    start = text.find("{")
    if start == -1:
        return _FALLBACK
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    return _FALLBACK


def build_judge_chain(model_id: str = DEFAULT_JUDGE_MODEL):
    """
    Load a local HuggingFace model and return an LCEL judge chain.

    Returns (chain, resources_to_free) so the caller can release GPU memory
    after evaluation:
        chain, resources = build_judge_chain()
        ...
        resources.clear(); gc.collect(); torch.cuda.empty_cache()

    Hardware selection:
      - CUDA available → 4-bit quantisation (BitsAndBytesConfig), ~1 GB VRAM
      - CPU only       → float32, no quantisation (~6 GB RAM for a 1.5B model)

    The LCEL chain is:
        ChatPromptTemplate
            | ChatHuggingFace  (applies the model's built-in chat template)
            | StrOutputParser  (extracts the string content from AIMessage)
            | RunnableLambda(_extract_json)  (parses JSON, handles messy output)
    """
    print(f"Loading judge model: {model_id}")

    if torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False,
        return_full_text=False,  # only the generated tokens, not the prompt
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    chat_model = ChatHuggingFace(llm=llm, verbose=False)

    chain = _PROMPT | chat_model | StrOutputParser() | RunnableLambda(_extract_json)
    return chain, [model, tokenizer]
