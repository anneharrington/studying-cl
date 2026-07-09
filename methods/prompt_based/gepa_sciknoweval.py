import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


class SciKnowEval(dspy.Signature):
    """Answer the scientific chemistry question. For multiple choice, respond with just the letter (A, B, C, or D). For other questions, give a direct answer without explanation."""

    question = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """Build an unoptimized chain-of-thought program for SciKnowEval."""
    return dspy.ChainOfThought(SciKnowEval)
