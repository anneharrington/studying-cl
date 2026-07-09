import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


class HoVer(dspy.Signature):
    """Determine whether the given claim is SUPPORTED or NOT_SUPPORTED based on your knowledge."""

    claim = dspy.InputField()
    label = dspy.OutputField(desc="SUPPORTED or NOT_SUPPORTED")


def build_program():
    """Build an unoptimized chain-of-thought program for HoVer."""
    return dspy.ChainOfThought(HoVer)
