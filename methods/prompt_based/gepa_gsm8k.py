import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


class GSM8K(dspy.Signature):
    """Solve the math word problem step by step. Show your work, then give the final numeric answer on its own line after ####."""

    question = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """Build an unoptimized chain-of-thought program for GSM8K."""
    return dspy.ChainOfThought(GSM8K)
