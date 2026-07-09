import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


class FinQA(dspy.Signature):
    """Answer the financial question using the provided narrative context and table. Think step by step, compute the numeric answer, and give it as a single number after "Answer:"."""

    question = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """Build an unoptimized chain-of-thought program for FinQA."""
    return dspy.ChainOfThought(FinQA)
