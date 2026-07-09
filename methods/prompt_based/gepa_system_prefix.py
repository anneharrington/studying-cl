"""GEPA "system-prefix" mode — keep GEPA's natural mutable slot (the system
message) but strip the DSPy [[ ## field ## ]] markers that conflict with
task-specific output formats (notably the SDPO Action format on tooluse,
which causes a 0% baseline under stock ChatAdapter).

Wire format:
    system: {evolved instructions}    ← GEPA mutates this, same as stock GEPA
    user:   {raw question}            ← no field markers, bare prompt

Compare to:
    stock GEPA: system has instructions + field schema + [[ ## answer ## ]];
                user wraps question in [[ ## question ## ]] markers.
                On tooluse the markers tell the model to "fill the answer
                field" and it fills with a fake tool-output JSON instead of
                emitting Action: / Action Input: → 0% baseline.
    raw-prefix: no system message at all, single user message.
                GEPA loses its natural mutable slot — can't improve.

This variant restores GEPA's native system slot AND restores bare-baseline
behavior (~56 on tooluse, matching ACE/OE), so GEPA can both start at the
model's natural ceiling AND improve from there via reflection.

ACE and OpenEvolve remain unchanged.
"""

from __future__ import annotations

from typing import Any

import dspy

from cl.tasks import TASK_REGISTRY


class SystemPrefixAdapter(dspy.adapters.ChatAdapter):
    """Format: system = evolved instructions, user = raw input. No markers.

    Inherits ChatAdapter so DSPy prediction tracking works (GEPA's reflection
    needs `dspy.history` traces). Only `format` and `parse` are overridden.
    """

    def format(self, signature, demos, inputs):
        instructions = signature.instructions or ""
        input_field_names = list(signature.input_fields.keys())
        if input_field_names:
            primary_in = input_field_names[0]
            content_in = str(inputs.get(primary_in, ""))
        else:
            content_in = ""

        msgs = []
        if instructions.strip():
            msgs.append({"role": "system", "content": instructions})
        msgs.append({"role": "user", "content": content_in})
        return msgs

    def parse(self, signature, completion, _parse_values=True):
        out_field_names = list(signature.output_fields.keys())
        primary_out = out_field_names[0] if out_field_names else "answer"
        return {primary_out: completion}


class SystemPrefixModule(dspy.Module):
    """Wraps dspy.Predict with SystemPrefixAdapter active during forward().

    `self.predict.signature.instructions` is the GEPA-mutable surface —
    same as stock GEPA — but the LM call uses our adapter so the wire format
    is bare system + bare user instead of DSPy's structured-fields format.
    """

    def __init__(self, signature: type[dspy.Signature]):
        super().__init__()
        self.predict = dspy.Predict(signature)
        self._adapter = SystemPrefixAdapter()

    def forward(self, **inputs: Any) -> dspy.Prediction:
        with dspy.context(adapter=self._adapter):
            return self.predict(**inputs)


def build_program_system_prefix(task_name: str) -> dspy.Module:
    """Return a SystemPrefixModule seeded with the task's instruction.

    Seed precedence: `default_instruction` if non-empty, else `system` from
    TASK_REGISTRY. For fin/bio/tool the `default_instruction` is intentionally
    empty (under raw-prefix wiring the SDPO system prompt sat frozen in
    `system`); under system-prefix wiring we surface that same SDPO prompt as
    the GEPA-mutable seed so baseline behavior matches RL.
    """
    info = TASK_REGISTRY[task_name]
    default_instruction = info.get("default_instruction", "") or ""
    if not default_instruction.strip():
        default_instruction = info.get("system", "") or ""
    template = info.get("template", "") or ""
    template_residue = template.replace("{question}", "").strip()

    initial = default_instruction
    # Only merge in template residue when it doesn't still contain unfilled
    # placeholders (e.g. finance_yr's template uses {filing_text} not
    # {question}, so the residue is the whole template — that's an
    # OpenEvolve-shape artifact, not a GEPA format hint).
    if template_residue and "{" not in template_residue:
        initial = f"{default_instruction}\n\n{template_residue}".strip()

    sig = type(
        "SystemPrefixSignature",
        (dspy.Signature,),
        {
            "__doc__": initial,
            "question": dspy.InputField(),
            "answer": dspy.OutputField(),
        },
    )
    return SystemPrefixModule(sig)
