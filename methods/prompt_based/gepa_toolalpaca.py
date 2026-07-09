import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


class ToolAlpaca(dspy.Signature):
    """Given API documentation and a user request, determine the correct sequence of API function calls. For each call, output the function name and its JSON parameters in this format:
Action: <function_name>
Action_Input: <json_parameters>

Separate multiple calls with a blank line."""

    question = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """Build an unoptimized chain-of-thought program for ToolAlpaca."""
    return dspy.ChainOfThought(ToolAlpaca)
