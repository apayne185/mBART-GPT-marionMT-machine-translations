"""
LLM-as-Judge evaluation chain built with LangChain LCEL.

Composes three LangChain primitives using the pipe operator:
    ChatPromptTemplate | ChatAnthropic | JsonOutputParser

Scores each machine translation on three linguistic dimensions:
  - fluency:   grammatical correctness and naturalness in German
  - adequacy:  preservation of source meaning without omissions or distortions
  - style:     appropriateness of register and word choice

Usage:
    chain = build_judge_chain()
    result = chain.invoke({"source": "...", "translation": "..."})
    # result = {"fluency": 8, "adequacy": 9, "style": 7, "overall": 8.0, "comment": "..."}

    # Parallel batch evaluation (max_concurrency limits simultaneous API calls):
    results = chain.batch(inputs, config={"max_concurrency": 4})
"""

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

_SYSTEM = (
    "You are an expert German linguist and machine translation evaluator. "
    "You will be given an English source sentence and a German translation produced by an MT system. "
    "Evaluate the translation strictly and objectively. "
    "Return only valid JSON — no markdown fences, no explanation outside the JSON object."
)

_HUMAN = """\
Source (English): {source}
Translation (German): {translation}

Score this translation on three dimensions (each 1–10):
- fluency:   Is the German grammatically correct and natural-sounding?
- adequacy:  Does the translation convey the full meaning of the source without omissions or distortions?
- style:     Is the word choice idiomatic, appropriately formal, and natural for a native speaker?

Set "overall" to the arithmetic mean of the three scores (rounded to one decimal place).
Add a one-sentence "comment" explaining the most notable strength or weakness.

Return exactly this JSON object and nothing else:
{{"fluency": <int 1-10>, "adequacy": <int 1-10>, "style": <int 1-10>, "overall": <float>, "comment": "<string>"}}"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])


def build_judge_chain(model: str = "claude-haiku-4-5-20251001"):
    """
    Build and return the LCEL judge chain.

    The chain is:
        ChatPromptTemplate | ChatAnthropic(temperature=0) | JsonOutputParser()

    temperature=0 ensures deterministic, consistent scoring across runs.
    claude-haiku-4-5-20251001 balances cost (cheap) and quality (strong instruction-following).
    """
    llm = ChatAnthropic(model=model, temperature=0)
    return _PROMPT | llm | JsonOutputParser()
