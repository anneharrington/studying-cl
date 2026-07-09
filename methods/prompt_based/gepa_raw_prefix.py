"""GEPA "raw prefix" mode — make GEPA evolve a single user-message prefix
that gets prepended to the question, with NO DSPy structured-output scaffold.

The resulting LM call structurally matches OpenEvolve's:
    messages=[{"role": "user", "content": f"{evolved_prefix}\\n\\n{question}"}]

Crucially, this implementation uses dspy.Predict + a custom Adapter so that
DSPy still records the prediction trace (inputs, completion, output) — which
GEPA's reflection mechanism needs to inspect failures and propose new
instructions. A pure manual `lm(messages=...)` bypass would break GEPA's
reflection ("No valid predictions found for any module" exception).
"""

from __future__ import annotations

from typing import Any, Dict, List

import dspy

from cl.tasks import TASK_REGISTRY, get_system_prompt


class RawPrefixAdapter(dspy.adapters.ChatAdapter):
    """Custom DSPy adapter: format → (optional system + ) single user message;
    parse → whole completion as `answer`.

    Inherits from ChatAdapter so it stays compatible with DSPy's prediction
    tracking and history mechanism (which GEPA's reflection needs), but
    overrides `format` and `parse` to produce / accept plain text rather than
    the [[ ## field ## ]] scaffold.

    `system_text`, when set, is sent as a separate role=system message before
    the user message — matches the training-time chat structure for tasks
    like FinQA / sciknoweval_bio whose baselines collapse without it.
    """

    def __init__(self, system_text: str | None = None):
        super().__init__()
        self._system_text = system_text or None

    def format(self, signature, demos, inputs):
        instructions = signature.instructions or ""
        input_field_names = list(signature.input_fields.keys())
        if input_field_names:
            primary = input_field_names[0]
            content_in = str(inputs.get(primary, ""))
        else:
            content_in = ""
        body = f"{instructions}\n\n{content_in}".strip() if instructions else content_in
        messages = []
        if self._system_text:
            messages.append({"role": "system", "content": self._system_text})
        messages.append({"role": "user", "content": body})
        return messages

    def parse(self, signature, completion, _parse_values=True):
        out_field_names = list(signature.output_fields.keys())
        primary = out_field_names[0] if out_field_names else "answer"
        return {primary: completion}


class RawPrefixModule(dspy.Module):
    """Wraps dspy.Predict with the RawPrefixAdapter active during forward().

    `self.predict.signature.instructions` remains the GEPA-mutable surface;
    the actual LM call uses our custom adapter so the messages-list is a
    single user message structurally identical to OpenEvolve (plus an
    optional system role for tasks that need it).
    """

    def __init__(self, signature: type[dspy.Signature], system_text: str | None = None):
        super().__init__()
        self.predict = dspy.Predict(signature)
        self._adapter = RawPrefixAdapter(system_text=system_text)

    def forward(self, **inputs: Any) -> dspy.Prediction:
        with dspy.context(adapter=self._adapter):
            return self.predict(**inputs)


def build_program_raw_prefix(task_name: str, seed_mode: str = "default") -> dspy.Module:
    """Return a RawPrefixModule with empty initial instructions.

    For tooluse (default_instruction=""), the initial prompt sent to the LM
    is just the question. GEPA's reflection LM may then propose non-empty
    instructions to improve over the empty baseline.

    For tasks with a `system` field in TASK_REGISTRY (FinQA, sciknoweval_bio),
    that string is sent as a separate role=system message — matches the
    training-time prompt structure exactly.

    `seed_mode="vanilla"` swaps `system` → `vanilla_system` (see
    cl.tasks.get_system_prompt) — the barebones ablation experiment.
    """
    info = TASK_REGISTRY[task_name]
    default_instruction = info.get("default_instruction", "") or ""
    template = info.get("template", "") or ""
    template_residue = template.replace("{question}", "").strip()
    system_text = get_system_prompt(task_name, mode=seed_mode)

    initial = default_instruction
    if template_residue:
        initial = f"{default_instruction}\n\n{template_residue}".strip()

    sig = type(
        "RawPrefixSignature",
        (dspy.Signature,),
        {
            "__doc__": initial,
            "question": dspy.InputField(),
            "answer": dspy.OutputField(),
        },
    )
    return RawPrefixModule(sig, system_text=system_text)
