import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


class ToolUse(dspy.Signature):
    """Use the available tools described in the question to answer the user's request."""
    # Intentionally minimal — the SDPO prompt already contains the full
    # "Use the following format: Thought: ... Action: ... Action Input: ..."
    # block. Adding our own format hint with `Action_Input` (underscore)
    # conflicted with the SDPO format (`Action Input` with space) and
    # measurably hurt qwen3-no-thinking. GEPA can re-introduce format
    # guidance during reflection if it's actually helpful.

    question = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """Build an unoptimized chain-of-thought program for tooluse."""
    return dspy.ChainOfThought(ToolUse)
