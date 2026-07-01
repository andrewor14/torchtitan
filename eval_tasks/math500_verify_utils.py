"""Custom MATH-500 scoring via math_verify (antlr-independent).

The stock lm_eval minerva_math / hendrycks_math tasks import sympy's
parse_latex, which asserts antlr4-python3-runtime==4.11. hydra 1.3.2 (used by
verl) pins antlr 4.9, so that path is unusable in this env. math_verify (via
latex2sympy2_extended) bundles its own parser and extracts \\boxed{} answers
natively, which matches how the chat-tuned model formats its responses.
"""

import datasets
from math_verify import parse, verify


def doc_to_text(doc: dict) -> str:
    # Plain problem prompt; --apply_chat_template wraps it in the model's
    # chat format. Asking for \boxed{} matches the SFT/RL answer style.
    return (
        "Solve the following math problem. Put your final answer in "
        "\\boxed{}.\n\n" + doc["problem"]
    )


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    candidate = results[0]
    # Gold answer: MATH-500 provides a clean "answer" field; fall back to the
    # full solution (which contains \boxed{}) if absent.
    gold_raw = doc.get("answer") or doc.get("solution", "")
    try:
        gold = parse("\\boxed{" + gold_raw + "}") if doc.get("answer") else parse(gold_raw)
        target = parse(candidate)
        correct = 1 if verify(gold=gold, target=target) else 0
    except Exception:
        correct = 0
    return {"math_verify": correct}
