"""Centralized task registry.

Adding a new task requires:
1. Creating cl/evals/<task>.py with load_<task>, load_<task>_raw, <task>_metric
2. Creating methods/prompt_based/gepa_<task>.py with build_program()
3. Creating methods/prompt_based/openevolve_<task>/ with evaluator.py, initial_prompt.txt
4. Adding an entry to TASK_REGISTRY below

All runner scripts, plotting, and configs reference tasks by name from this registry.
"""

from pathlib import Path

# Lazy imports — resolved at access time to avoid importing everything at startup.
# Each entry maps a task name to its components.

TASK_REGISTRY = {
    "hotpotqa": {
        # GEPA components
        "loader": "cl.evals.hotpotqa:load_hotpotqa",
        "loader_raw": "cl.evals.hotpotqa:load_hotpotqa_raw",
        "metric": "cl.evals.hotpotqa:hotpotqa_metric",
        "gepa_build_program": "methods.prompt_based.gepa_method:build_program",
        # OpenEvolve components (relative to project root)
        "openevolve_evaluator": "methods/prompt_based/openevolve_hotpotqa/evaluator.py",
        "openevolve_prompt": "methods/prompt_based/openevolve_hotpotqa/initial_prompt.txt",
        # OpenEvolve sequential: template + default instruction
        "template": (
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            'Think step by step, then provide your final answer on its own line after "Answer:".'
        ),
        "default_instruction": "Answer the question based on the provided context paragraphs.",
    },
    "ifeval": {
        "loader": "cl.evals.ifeval:load_ifeval",
        "loader_raw": "cl.evals.ifeval:load_ifeval_raw",
        "metric": "cl.evals.ifeval:ifeval_metric",
        "gepa_build_program": "methods.prompt_based.gepa_ifeval:build_program",
        "openevolve_evaluator": "methods/prompt_based/openevolve_ifeval/evaluator.py",
        "openevolve_prompt": "methods/prompt_based/openevolve_ifeval/initial_prompt.txt",
        "template": "{prompt}",
        "default_instruction": (
            "Follow all instructions in the prompt below precisely and completely. "
            "Pay careful attention to every formatting requirement, length constraint, "
            "keyword inclusion, and content specification."
        ),
    },
    "hover": {
        "loader": "cl.evals.hover:load_hover",
        "loader_raw": "cl.evals.hover:load_hover_raw",
        "metric": "cl.evals.hover:hover_metric",
        "gepa_build_program": "methods.prompt_based.gepa_hover:build_program",
        "openevolve_evaluator": "methods/prompt_based/openevolve_hover/evaluator.py",
        "openevolve_prompt": "methods/prompt_based/openevolve_hover/initial_prompt.txt",
        "template": (
            "Claim: {claim}\n\n"
            'Think step by step, then provide your final verdict on its own line '
            'after "Label:". The label must be exactly SUPPORTED or NOT_SUPPORTED '
            '(no other strings, no quotes, no punctuation).'
        ),
        "default_instruction": (
            'Determine whether the following claim is factually correct. Respond with '
            'exactly "SUPPORTED" if the claim is true, or "NOT_SUPPORTED" if the claim '
            'is false or unverifiable.'
        ),
    },
    "sciknoweval": {
        "loader": "cl.evals.sciknoweval:load_sciknoweval",
        "loader_raw": "cl.evals.sciknoweval:load_sciknoweval_raw",
        "metric": "cl.evals.sciknoweval:sciknoweval_metric",
        "gepa_build_program": "methods.prompt_based.gepa_sciknoweval:build_program",
        "openevolve_evaluator": "methods/prompt_based/openevolve_sciknoweval/evaluator.py",
        "openevolve_prompt": "methods/prompt_based/openevolve_sciknoweval/initial_prompt.txt",
        "template": "{question}\n\n"
                    'Think step by step, then provide your final answer on its own line '
                    'after "Answer:". For multiple-choice questions, the answer line must '
                    'contain only the single letter (A, B, C, or D). For other questions, '
                    'give a direct, concise answer.',
        "default_instruction": (
            "Answer the scientific chemistry question. For multiple choice, respond "
            "with just the letter (A, B, C, or D). For other questions, give a direct answer."
        ),
    },
    "gsm8k": {
        "loader": "cl.evals.gsm8k:load_gsm8k",
        "loader_raw": "cl.evals.gsm8k:load_gsm8k_raw",
        "metric": "cl.evals.gsm8k:gsm8k_metric",
        "gepa_build_program": "methods.prompt_based.gepa_gsm8k:build_program",
        "openevolve_evaluator": "methods/prompt_based/openevolve_gsm8k/evaluator.py",
        "openevolve_prompt": "methods/prompt_based/openevolve_gsm8k/initial_prompt.txt",
        "template": "{question}\n\n"
                    'Show your reasoning step by step, then give the final numeric answer after "####".',
        "default_instruction": (
            "Solve the math word problem step by step. Show your work, "
            "then give the final numeric answer on its own line after ####."
        ),
    },
    "livebench_math": {
        "loader": "cl.evals.livebench_math:load_livebench_math",
        "loader_raw": "cl.evals.livebench_math:load_livebench_math_raw",
        "metric": "cl.evals.livebench_math:livebench_math_metric",
        "gepa_build_program": "methods.prompt_based.gepa_livebench_math:build_program",
        "openevolve_evaluator": "methods/prompt_based/openevolve_livebench_math/evaluator.py",
        "openevolve_prompt": "methods/prompt_based/openevolve_livebench_math/initial_prompt.txt",
        "template": (
            "{question}\n\n"
            "Think step by step, then give your final answer at the very end of "
            "your response in the exact format specified by the question (a single "
            "three-digit number like 042, or a single multiple-choice letter like B)."
        ),
        "default_instruction": (
            "Solve the competition math problem. Think step by step, then give "
            "your final answer at the very end of your response in the format "
            "specified by the question."
        ),
    },
    "sciknoweval_bio": {
        "loader": "cl.evals.sciknoweval_bio:load_sciknoweval_bio",
        "loader_raw": "cl.evals.sciknoweval_bio:load_sciknoweval_bio_raw",
        "metric": "cl.evals.sciknoweval_bio:sciknoweval_bio_metric",
        "gepa_build_program": "methods.prompt_based.gepa_sciknoweval_bio:build_program",
        "openevolve_evaluator": "methods/prompt_based/openevolve_sciknoweval_bio/evaluator.py",
        "openevolve_prompt": "methods/prompt_based/openevolve_sciknoweval_bio/initial_prompt.txt",
        # Match training prompt structure: system role carries the
        # <reasoning>/<answer> XML format prompt; user message is the SDPO
        # data file's pre-formatted prompt (question + choices + "Please reason
        # step by step."). Wrapping with the legacy Answer:-style scaffold
        # tanked the baseline 33.75 → 20 (same scaffold-tax pattern as FinQA).
        "system": (
            "\nGiven a question and four options, please select the right answer. "
            "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n"
            "<answer>\n...\n</answer>\n\nFor the answer, only output the letter "
            "corresponding to the correct option (A, B, C, or D), and nothing else. "
            "Do not restate the answer text. For example, if the answer is \"A\", "
            "just output:\n<answer>\nA\n</answer>\n"
        ),
        # Vanilla ablation: strips the XML format spec and the worked example.
        # The MCQ extractor's bare-letter fallback ([A-D] at line start) is the
        # only remaining anchor — measures format-correction headroom.
        "vanilla_system": "Answer the biology question.",
        "template": "{question}",
        "default_instruction": "",
    },
    "tooluse": {
        "loader": "cl.evals.tooluse:load_tooluse",
        "loader_raw": "cl.evals.tooluse:load_tooluse_raw",
        "metric": "cl.evals.tooluse:tooluse_metric",
        "gepa_build_program": "methods.prompt_based.gepa_tooluse:build_program",
        "openevolve_evaluator": "methods/prompt_based/openevolve_tooluse/evaluator.py",
        "openevolve_prompt": "methods/prompt_based/openevolve_tooluse/initial_prompt.txt",
        "template": "{question}",
        # default_instruction intentionally empty: the SDPO `prompt` already
        # has the full tool documentation and the "Use the following format:
        # ... Action: ... Action Input: ..." block. Adding a separate
        # default_instruction that uses `Action_Input` (underscore) conflicts
        # with the SDPO prompt's `Action Input` (space) format hint and
        # measurably hurts qwen3-no-thinking. Matching the verl training pipeline
        # gives strict baseline ≈ 0.58 (SDPO prompt verbatim + strict scoring).
        "default_instruction": "",
    },
    "finqa": {
        "loader": "cl.evals.finqa:load_finqa",
        "loader_raw": "cl.evals.finqa:load_finqa_raw",
        "metric": "cl.evals.finqa:finqa_metric",
        "gepa_build_program": "methods.prompt_based.gepa_finqa:build_program",
        "openevolve_evaluator": "methods/prompt_based/openevolve_finqa/evaluator.py",
        "openevolve_prompt": "methods/prompt_based/openevolve_finqa/initial_prompt.txt",
        # Match training prompt structure exactly: system role carries the
        # `\boxed{}` reasoning instruction, user message is the raw flare-finqa
        # question. Wrapping `{question}` with a verbose Answer:-style scaffold
        # cost ~8pts on the Qwen3-8B baseline (parallel to the DSPy
        # [[ ## answer ## ]] scaffold tax that tanked tooluse 56→0).
        "system": (
            "You are a careful financial-reasoning assistant. Read the context, "
            "reason step by step, and put ONLY the final numerical answer inside "
            "\\boxed{}. Do not include units or punctuation inside the box."
        ),
        # Vanilla ablation: drops role framing AND \boxed{} contract. Extractor
        # falls back to last-number-in-text — fragile but nonzero.
        "vanilla_system": "Answer the financial question.",
        "template": "{question}",
        "default_instruction": "",
    },
    "toolalpaca": {
        "loader": "cl.evals.toolalpaca:load_toolalpaca",
        "loader_raw": "cl.evals.toolalpaca:load_toolalpaca_raw",
        "metric": "cl.evals.toolalpaca:toolalpaca_metric",
        "gepa_build_program": "methods.prompt_based.gepa_toolalpaca:build_program",
        "openevolve_evaluator": "methods/prompt_based/openevolve_toolalpaca/evaluator.py",
        "openevolve_prompt": "methods/prompt_based/openevolve_toolalpaca/initial_prompt.txt",
        "template": "{question}",
        "default_instruction": (
            "Given API documentation and a user request, determine the correct sequence "
            "of API function calls. For each call, output the function name and JSON "
            "parameters as: Action: <name>\\nAction_Input: <json>"
        ),
    },
}

# Sentiment10K is registered as one task per filing year so the standard
# sequential runner can iterate years exactly the way it iterates tasks. The
# loader receives `year_filter` from the per-task dataset block, so each entry
# loads only that year's filings. Years <2018 and 2023 omitted from defaults
# because they have <4 docs after `status=ok` filtering — add more if your
# rebuilt sentiment.json has them.
_SENTIMENT10K_BASE = {
    "loader": "cl.evals.sentiment10k:load_sentiment10k",
    "loader_raw": "cl.evals.sentiment10k:load_sentiment10k_raw",
    "metric": "cl.evals.sentiment10k:sentiment10k_metric",
    "gepa_build_program": "methods.prompt_based.gepa_sentiment10k:build_program",
    "openevolve_evaluator": "methods/prompt_based/openevolve_sentiment10k/evaluator.py",
    "openevolve_prompt": "methods/prompt_based/openevolve_sentiment10k/initial_prompt.txt",
    "template": "{filing_text}\n\nReturn one token: up or down.",
    "default_instruction": (
        "Read the 10-K filing excerpt and classify the company's forward stock-movement "
        "sentiment. Output exactly one token: 'up' or 'down'."
    ),
}
for _year in (2018, 2019, 2020, 2021, 2022):
    TASK_REGISTRY[f"sentiment10k_{_year}"] = dict(_SENTIMENT10K_BASE)


# SealQA is registered analogously to sentiment10k: a base entry plus one
# per-effective-year subtask, so the standard sequential runner can iterate
# years exactly the way it iterates tasks. The loader receives `year_filter`
# from the per-task dataset block. Update the year list below after running
# `python cl/finance/build_data/build_sealQA.py` and inspecting which
# effective_years actually have ≥ a few fast-changing rows.
# finance_yr is the SDPO finance bundle's per-year sentiment task: 6 yearly
# splits (2015..2020) of 10-K forward-direction up/down classification, read
# directly from temporal_shift/finance_data_bundle/data/*.parquet so GEPA
# scores on the same 50 val rows the RL methods (SFT/SDFT/GRPO/SDPO) used.
# Each year is registered as its own task entry so the standard sequential
# runner iterates years exactly the way it iterates tasks (the loader
# receives `year_filter` from the per-task dataset block).
_FINANCE_YR_BASE = {
    "loader": "cl.evals.finance_yr:load_finance_yr",
    "loader_raw": "cl.evals.finance_yr:load_finance_yr_raw",
    "metric": "cl.evals.finance_yr:finance_yr_metric",
    "gepa_build_program": "methods.prompt_based.gepa_finance_yr:build_program",
    # OpenEvolve mirror not yet implemented; paths kept as placeholders so the
    # registry shape stays uniform. GEPA-only runs never resolve these.
    "openevolve_evaluator": "methods/prompt_based/openevolve_finance_yr/evaluator.py",
    "openevolve_prompt": "methods/prompt_based/openevolve_finance_yr/initial_prompt.txt",
    "template": "{filing_text}\n\nReturn one token: up or down.",
    # Verbatim from temporal_shift/finance_data_bundle/data/finance_manifest.json
    # `system_prompt` (also baked into gepa_finance_yr.py's Signature.__doc__) so
    # baseline GEPA under use_system_prefix sees the same prompt the RL methods
    # were trained against.
    "default_instruction": (
        "You are a financial analyst. Given a 10-K filing excerpt, predict the "
        "forward stock direction over the next 30 days after the filing date. "
        "Return ONE word ONLY: up or down. No explanation, no quotation marks, "
        "no extra words.\n"
        "Examples:\n"
        "  ... -> up\n"
        "  ... -> down"
    ),
}
for _year in (2015, 2016, 2017, 2018, 2019, 2020):
    TASK_REGISTRY[f"finance_yr_{_year}"] = dict(_FINANCE_YR_BASE)


# temporalwiki is the SDPO TemporalWiki bundle's drift Q&A task: 3 sequential
# drift slices (s1/s2/s3, Nov 2025 → Feb 2026 chronological pairs) plus a
# `stable` eval-only probe whose 50 (subject, relation) keys never overlap
# the drift slices. Read directly from temporal_shift/temporalwiki_data/
# cl_drift_data/*.parquet so GEPA scores on the same 50 val rows per slice
# the RL methods evaluated on. Task names match the SDPO data_source values
# verbatim (temporalwiki_drift_s<i>, temporalwiki_stable) so the SDPO-format
# val-core/<source>/acc/mean@N keys map 1:1.
_TEMPORALWIKI_BASE = {
    "loader": "cl.evals.temporalwiki:load_temporalwiki",
    "loader_raw": "cl.evals.temporalwiki:load_temporalwiki_raw",
    "metric": "cl.evals.temporalwiki:temporalwiki_metric",
    "gepa_build_program": "methods.prompt_based.gepa_temporalwiki:build_program",
    "openevolve_evaluator": "methods/prompt_based/openevolve_temporalwiki/evaluator.py",
    "openevolve_prompt": "methods/prompt_based/openevolve_temporalwiki/initial_prompt.txt",
    "template": "{question}",
    # Verbatim from temporal_shift/temporalwiki_data/cl_drift_data/manifest.json
    # `system_prompt` (also baked into gepa_temporalwiki.py's Signature.__doc__)
    # so baseline GEPA under use_system_prefix sees the same prompt the RL
    # methods were trained against.
    "default_instruction": (
        "You are answering factual knowledge questions about Wikipedia entities. "
        "Given a subject and a relation, output ONLY the object value as a short "
        "plain-text string. No explanation, no quotation marks, no markup, no "
        "extra words.\n"
        "Examples:\n"
        "  Marshal Yanda educated at -> University of Iowa\n"
        "  Hans Zimmer spouse -> Suzanne Zimmer\n"
        "  Ho Chi Minh City contains administrative territorial entity -> District 7"
    ),
}
for _slice in ("s1", "s2", "s3"):
    TASK_REGISTRY[f"temporalwiki_drift_{_slice}"] = dict(_TEMPORALWIKI_BASE)
TASK_REGISTRY["temporalwiki_stable"] = dict(_TEMPORALWIKI_BASE)


_SEALQA_BASE = {
    "loader": "cl.evals.sealqa:load_sealqa",
    "loader_raw": "cl.evals.sealqa:load_sealqa_raw",
    "metric": "cl.evals.sealqa:sealqa_metric",
    "gepa_build_program": "methods.prompt_based.gepa_sealqa:build_program",
    "openevolve_evaluator": "methods/prompt_based/openevolve_sealqa/evaluator.py",
    "openevolve_prompt": "methods/prompt_based/openevolve_sealqa/initial_prompt.txt",
    "template": "{question}",
    "default_instruction": (
        "Answer the question using only the provided context documents. "
        'Return only the final answer on its own line after "Answer:".'
    ),
}
# Base name for non-yearly runs (whole fast-changing partition); per-year
# subtasks below for the yearly sequential pattern.
TASK_REGISTRY["sealqa"] = dict(_SEALQA_BASE)
for _year in (2022, 2023, 2024, 2025, 2026):
    TASK_REGISTRY[f"sealqa_{_year}"] = dict(_SEALQA_BASE)


def get_system_prompt(task_name, mode="default"):
    """Return the role=system message for `task_name` under `mode`.

    mode="default" returns the standard `system` field (current behavior).
    mode="vanilla" returns `vanilla_system` if set, else None (no fallback to
    `system` — vanilla mode means strip the scaffold the optimizer can't see).
    Tasks without a `system` field (e.g. tooluse) return None in either mode.
    """
    entry = TASK_REGISTRY[task_name]
    if mode == "vanilla":
        return entry.get("vanilla_system")
    return entry.get("system")


def _import(dotted_path):
    """Import a callable from a 'module.path:attribute' string."""
    module_path, attr = dotted_path.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def get_task(name):
    """Return a resolved task dict with callable loader/metric/build_program.

    Raises KeyError if the task name is not registered.
    """
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}")

    entry = TASK_REGISTRY[name]
    return {
        "loader": _import(entry["loader"]),
        "loader_raw": _import(entry["loader_raw"]),
        "metric": _import(entry["metric"]),
        "build_program": _import(entry["gepa_build_program"]),
        "openevolve_evaluator": entry["openevolve_evaluator"],
        "openevolve_prompt": entry["openevolve_prompt"],
        "template": entry.get("template", ""),
        "default_instruction": entry.get("default_instruction", ""),
    }


def get_gepa_tasks(task_names):
    """Return {name: {loader, metric, build_program}} for GEPA scripts."""
    return {
        name: {
            "loader": _import(TASK_REGISTRY[name]["loader"]),
            "metric": _import(TASK_REGISTRY[name]["metric"]),
            "build_program": _import(TASK_REGISTRY[name]["gepa_build_program"]),
        }
        for name in task_names
    }


def get_openevolve_tasks(task_names, project_root=None):
    """Return {name: {loader_raw, evaluator_path, prompt_path, template, default_instruction}}."""
    if project_root is None:
        project_root = str(Path(__file__).resolve().parent.parent)
    return {
        name: {
            "loader_raw": _import(TASK_REGISTRY[name]["loader_raw"]),
            "evaluator": str(Path(project_root) / TASK_REGISTRY[name]["openevolve_evaluator"]),
            "initial_prompt": str(Path(project_root) / TASK_REGISTRY[name]["openevolve_prompt"]),
            "template": TASK_REGISTRY[name].get("template", ""),
            "default_instruction": TASK_REGISTRY[name].get("default_instruction", ""),
        }
        for name in task_names
    }
