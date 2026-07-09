"""GEPA program builder for temporalwiki drift Q&A.

Default instructions are the SDPO TemporalWiki bundle's system prompt verbatim
(temporalwiki_data/cl_drift_data/manifest.json:system_prompt) so the GEPA
baseline is comparable to the RL step-0 (val_before_train) eval on the same
50 val rows per slice.
"""
import dspy

from methods.prompt_based.gepa_method import configure_lms, run_gepa  # noqa: F401


_BUNDLE_TEMPORALWIKI_SYSTEM_PROMPT = (
    "You are answering factual knowledge questions about Wikipedia entities. "
    "Given a subject and a relation, output ONLY the object value as a short "
    "plain-text string. No explanation, no quotation marks, no markup, no "
    "extra words. If multiple values are valid, pick the most canonical one.\n"
    "Examples:\n"
    "  Marshal Yanda educated at -> University of Iowa\n"
    "  Hans Zimmer spouse -> Suzanne Zimmer\n"
    "  Ho Chi Minh City contains administrative territorial entity -> District 7"
)


class TemporalWiki(dspy.Signature):
    __doc__ = _BUNDLE_TEMPORALWIKI_SYSTEM_PROMPT

    question = dspy.InputField()
    answer = dspy.OutputField()


def build_program():
    """ChainOfThought program for temporalwiki. The metric strips <think> tags
    and chat tail before extraction, so CoT reasoning doesn't leak into the
    F1 calculation."""
    return dspy.ChainOfThought(TemporalWiki)
