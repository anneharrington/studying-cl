import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


class Sentiment10K(dspy.Signature):
    """Read the 10-K filing excerpt and classify the company's forward stock-movement sentiment. Reason briefly about the financial signals in the filing, then output exactly one token after "Answer:" — either "up" or "down"."""

    filing_text = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """Build an unoptimized chain-of-thought program for Sentiment10K."""
    return dspy.ChainOfThought(Sentiment10K)
