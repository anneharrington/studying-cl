"""GEPA "CoT user-prefix" mode — keep DSPy's CoT reasoning field and field
markers (so GEPA's reflection has improvement headroom), but move the
GEPA-mutable string from the system message into the user message so the
evolved target lands in the same slot ACE and OpenEvolve evolve.

Wire format becomes:
    system: "<field schema + reasoning + answer markers, NO task instructions>"
    user:   "<EVOLVED_INSTRUCTIONS>\n\n[[ ## question ## ]]\n<question>\n\n
             Respond with the corresponding output fields, starting with
             [[ ## reasoning ## ]] then [[ ## answer ## ]]..."

Compare to:
    raw_prefix: user-only, no CoT field, no markers — GEPA can't improve.
    stock GEPA: instructions in *system*, CoT field present — GEPA improves
                but the evolved slot doesn't match ACE/OE.

ACE/OE remain unchanged. Only GEPA changes shape.
"""

from __future__ import annotations

from typing import Any

import dspy

from cl.tasks import TASK_REGISTRY


class CoTUserPrefixAdapter(dspy.adapters.ChatAdapter):
    """ChatAdapter variant that puts signature.instructions at the start of the
    user message instead of inside the system message.

    Implementation: render with an empty-instructions clone of the signature
    so the system message contains only field schema, then prepend the real
    instructions to the first user message. Output parsing is inherited
    unchanged — the model still emits [[ ## reasoning ## ]] / [[ ## answer ## ]]
    markers, so DSPy's prediction tracking and GEPA's reflection both work.
    """

    def format(self, signature, demos, inputs):
        instructions = signature.instructions or ""
        scratch = signature.with_instructions("")
        msgs = super().format(scratch, demos, inputs)
        if instructions:
            for m in msgs:
                if m.get("role") == "user":
                    m["content"] = f"{instructions}\n\n{m['content']}"
                    break
        return msgs


class CoTUserPrefixModule(dspy.Module):
    """Wraps dspy.ChainOfThought with CoTUserPrefixAdapter active during forward().

    `self.cot.predict.signature.instructions` is the GEPA-mutable surface
    (same as stock GEPA), but at LM-call time the adapter relocates it into
    the user message so the evolved content lands in the same slot ACE/OE
    optimize.
    """

    def __init__(self, signature: type[dspy.Signature]):
        super().__init__()
        self.cot = dspy.ChainOfThought(signature)
        self._adapter = CoTUserPrefixAdapter()

    def forward(self, **inputs: Any) -> dspy.Prediction:
        with dspy.context(adapter=self._adapter):
            return self.cot(**inputs)


def build_program_cot_user_prefix(task_name: str) -> dspy.Module:
    """Return a CoTUserPrefixModule seeded with the task's default instruction."""
    info = TASK_REGISTRY[task_name]
    default_instruction = info.get("default_instruction", "") or ""
    template = info.get("template", "") or ""
    template_residue = template.replace("{question}", "").strip()

    initial = default_instruction
    if template_residue:
        initial = f"{default_instruction}\n\n{template_residue}".strip()

    sig = type(
        "CoTUserPrefixSignature",
        (dspy.Signature,),
        {
            "__doc__": initial,
            "question": dspy.InputField(),
            "answer": dspy.OutputField(),
        },
    )
    return CoTUserPrefixModule(sig)
