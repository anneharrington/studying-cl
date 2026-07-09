import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


class SealQA(dspy.Signature):
    """Answer the question using only the provided context documents. Reason briefly, then return only the final answer on its own line after "Answer:"."""

    question = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """Build an unoptimized chain-of-thought program for SealQA."""
    return dspy.ChainOfThought(SealQA)
