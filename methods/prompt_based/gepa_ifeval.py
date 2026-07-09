import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


class IFEval(dspy.Signature):
    """Follow all instructions in the prompt precisely and completely."""

    prompt = dspy.InputField()
    response = dspy.OutputField()


def build_program():
    """Build an unoptimized chain-of-thought program for IFEval."""
    return dspy.ChainOfThought(IFEval)
