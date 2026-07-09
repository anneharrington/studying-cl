"""GEPA program builder for finance_yr.

Default instructions are the SDPO finance bundle's system prompt verbatim
(finance_data_bundle/data/finance_manifest.json:system_prompt) so the GEPA
baseline is comparable to the RL step-0 (val_before_train) eval on the same
50 val rows.
"""
import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


_SDPO_FINANCE_SYSTEM_PROMPT = (
    "You are a financial analyst. Given a 10-K filing excerpt, predict the "
    "forward stock direction over the next 30 days after the filing date. "
    "Return ONE word ONLY: up or down. No explanation, no quotation marks, "
    "no extra words.\n"
    "Examples:\n"
    "  ... -> up\n"
    "  ... -> down"
)


class FinanceYr(dspy.Signature):
    __doc__ = _SDPO_FINANCE_SYSTEM_PROMPT

    filing_text = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """ChainOfThought program for finance_yr; CoT is harmless because the
    metric extracts the first up|down token after stripping the chat tail."""
    return dspy.ChainOfThought(FinanceYr)
