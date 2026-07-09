import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


class LiveBenchMath(dspy.Signature):
    """Solve the competition math problem. Think step by step, then give your final answer at the very end of your response in the format specified by the question."""

    question = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """Build an unoptimized chain-of-thought program for LiveBench math."""
    return dspy.ChainOfThought(LiveBenchMath)
