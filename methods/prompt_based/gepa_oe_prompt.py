"""GEPA fairness mode: build_program_oe_style(task_name).

Constructs a dspy.Predict program (no ChainOfThought reasoning scaffold)
whose signature instructions match OpenEvolve's baseline prompt for that
task — i.e. the task's `default_instruction` followed by the `template`,
exactly what `openevolve_runner.py:build_prompt` produces.

This lets a GEPA run start optimization from the same prompt OE starts
from, so comparison deltas aren't confounded by DSPy's structured-output
scaffolding inflating the gap.
"""

from __future__ import annotations

import dspy

from cl.tasks import TASK_REGISTRY


def build_program_oe_style(task_name: str) -> dspy.Module:
    """Return a dspy.Predict program seeded with OE's default prompt.

    The signature has a single InputField `question` and OutputField `answer`
    (matching the existing per-task GEPA signatures). The class docstring is
    the task's default_instruction concatenated with its template (with the
    `{question}` placeholder removed since `question` is already a structured
    input field in DSPy).
    """
    info = TASK_REGISTRY[task_name]
    default_instruction = info.get("default_instruction", "")
    template = info.get("template", "")
    # Strip {question} from the template since DSPy will render the question
    # input field separately. Keep the rest (format hints like "Answer:").
    template_residue = template.replace("{question}", "").strip()

    instruction = default_instruction
    if template_residue:
        instruction = f"{default_instruction}\n\n{template_residue}"

    # Define a signature class on the fly with the merged instruction as docstring.
    sig = type(
        "OEStyleSignature",
        (dspy.Signature,),
        {
            "__doc__": instruction,
            "question": dspy.InputField(),
            "answer": dspy.OutputField(),
        },
    )

    # Wrap dspy.Predict in a Module that exposes a `.predict` attribute, since
    # the existing gepa_runner code mutates `program.predict.signature` (a
    # convention from dspy.ChainOfThought-based programs).
    class OEStyleModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict(sig)

        def forward(self, **kwargs):
            return self.predict(**kwargs)

    return OEStyleModule()
