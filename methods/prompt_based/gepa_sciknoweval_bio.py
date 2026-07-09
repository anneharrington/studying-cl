import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


class SciKnowEvalBio(dspy.Signature):
    """Answer the scientific biology question. Respond with just the letter (A, B, C, or D) of the correct choice."""

    question = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """Build an unoptimized chain-of-thought program for sciknoweval-bio."""
    return dspy.ChainOfThought(SciKnowEvalBio)
