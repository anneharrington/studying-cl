"""Composable config loader.

Assembles a flat runner config from component files:
  - configs/models/<model>.yaml   — LM endpoints, parallelism
  - configs/tasks/<task>.yaml     — dataset paths, split sizes
  - configs/methods/<method>.yaml — optimization hyperparameters

Run configs can either be:
  1. Traditional flat YAML files (passed directly, no assembly needed)
  2. Composable YAML files with a "model" string referencing a model profile

Usage from run.py:
    cfg = load_config(args.config, method=args.method, model=args.model,
                      tasks=args.tasks, task=args.task)
"""

import copy
import uuid
from datetime import datetime
from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _unique_output_dir(base_dir):
    """Append a timestamp and short random ID to an output directory path.

    results/gepa_sequential → results/gepa_sequential_20260402_a3f1
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    short_id = uuid.uuid4().hex[:4]
    return f"{base_dir}_{stamp}_{short_id}"


def ensure_unique_output_dir(cfg):
    """Add timestamp+ID to cfg['output_dir'] if it doesn't already have one.

    Call this from runners that use traditional (non-composable) configs,
    so they also get unique output dirs. Safe to call multiple times —
    it detects if a timestamp suffix is already present.
    """
    import re
    output_dir = cfg.get("output_dir", "results/run")
    # Skip if already has a timestamp suffix (YYYYMMDD_HHMM_xxxx)
    if re.search(r"_\d{8}_\d{4}_[0-9a-f]{4}$", output_dir):
        return cfg
    cfg["output_dir"] = _unique_output_dir(output_dir)
    return cfg


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_model(model_name):
    """Load a model profile from configs/models/<name>.yaml."""
    path = CONFIGS_DIR / "models" / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Model profile not found: {path}")
    return _load_yaml(path)


def _resolve_task(task_name):
    """Load a task dataset config from configs/tasks/<name>.yaml."""
    path = CONFIGS_DIR / "tasks" / f"{task_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Task config not found: {path}")
    return _load_yaml(path)


def _resolve_method(method_name):
    """Load method defaults from configs/methods/<name>.yaml."""
    path = CONFIGS_DIR / "methods" / f"{method_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Method config not found: {path}")
    return _load_yaml(path)


def _strip_litellm_prefix(model_id):
    """Strip the 'openai/' prefix that DSPy/litellm needs but raw OpenAI clients don't.

    'openai/qwen/qwen3-8b' → 'qwen/qwen3-8b'
    'openai/@google/gemini-2.5-flash-lite' → '@google/gemini-2.5-flash-lite'
    'gemini/gemini-2.5-flash' → 'gemini/gemini-2.5-flash' (no change)
    """
    if model_id.startswith("openai/"):
        return model_id[len("openai/"):]
    return model_id


def _build_model_section(model_profile, method):
    """Build the 'model' section of a flat config from a model profile."""
    task_lm = model_profile["task_lm"]

    # OpenEvolve + ACE use the raw OpenAI client — strip litellm 'openai/' prefix
    if method in ("openevolve", "ace", "ace-minimal"):
        task_lm = _strip_litellm_prefix(task_lm)

    model_cfg = {
        "task_lm": task_lm,
        "api_key_env": model_profile.get("api_key_env", "PORTKEY_API_KEY"),
    }
    if "api_base" in model_profile:
        model_cfg["api_base"] = model_profile["api_base"]
    for key in ("thinking", "task_thinking", "reflection_thinking",
                "task_max_tokens", "reflection_max_tokens",
                "task_temperature", "reflection_temperature",
                "extra_body"):
        if key in model_profile:
            model_cfg[key] = copy.deepcopy(model_profile[key])

    # GEPA needs reflection_lm. reflection_api_key_env / reflection_api_base
    # are optional: set them in the model profile to run the reflector on a
    # different provider/account than the task LM (split-provider runs). When
    # absent, the reflector reuses the task LM's key/base (original behaviour).
    if method == "gepa" and "reflection_lm" in model_profile:
        model_cfg["reflection_lm"] = model_profile["reflection_lm"]
        for key in ("reflection_api_key_env", "reflection_api_base"):
            if key in model_profile:
                model_cfg[key] = copy.deepcopy(model_profile[key])

    return model_cfg


def _build_method_section(method, method_defaults, model_profile, overrides=None):
    """Build the method-specific section (gepa or openevolve)."""
    cfg = copy.deepcopy(method_defaults)

    if method == "gepa":
        cfg["num_threads"] = model_profile.get("num_threads", 2)
        cfg.setdefault("eval_num_threads", model_profile.get("eval_num_threads",
                                                              cfg["num_threads"]))
    elif method == "openevolve":
        # Inject the model's evolution LLM config
        if "openevolve_llm" in model_profile:
            cfg["llm"] = copy.deepcopy(model_profile["openevolve_llm"])

    if overrides:
        cfg.update(overrides)

    return cfg


def _apply_split_overrides(cfg, split_overrides):
    """Apply CLI --train-n/--val-n/--eval-n to every dataset block in cfg.

    Only non-None overrides are applied, so any of the three can be omitted.
    Works for both traditional flat configs (cfg['dataset']) and composable
    multi-task configs (cfg['tasks'][i]['dataset']).
    """
    if not split_overrides:
        return cfg
    keys = [(k, v) for k, v in split_overrides.items() if v is not None]
    if not keys:
        return cfg

    if isinstance(cfg.get("dataset"), dict):
        for k, v in keys:
            cfg["dataset"][k] = v

    for task_entry in cfg.get("tasks") or []:
        if isinstance(task_entry, dict) and isinstance(task_entry.get("dataset"), dict):
            for k, v in keys:
                task_entry["dataset"][k] = v

    return cfg


def load_config(config_path, method=None, model=None, tasks=None, task=None,
                split_overrides=None):
    """Load and assemble a runner config.

    Supports two modes:

    1. Traditional: config_path is a complete flat YAML → returned as-is
       (detected by presence of 'model' dict or 'dataset' dict in the file)

    2. Composable: config_path is a thin YAML with 'model: "<profile-name>"'
       string, and optionally 'tasks: [name, ...]'. CLI args --model and
       --tasks can override.

    CLI overrides (model, tasks, task) take precedence over file contents.
    split_overrides is a dict of {'train_n', 'val_n', 'eval_n'} (any may be
    None); non-None values are applied to every dataset block after assembly.
    """
    cfg = _load_yaml(config_path)

    # --- Detect traditional flat config ---
    # If 'model' is already a dict (not a string), it's a traditional config
    if isinstance(cfg.get("model"), dict):
        return _apply_split_overrides(cfg, split_overrides)
    # If it has 'dataset' as a dict with 'path', it's a traditional single-task config
    if isinstance(cfg.get("dataset"), dict) and "path" in cfg.get("dataset", {}):
        return _apply_split_overrides(cfg, split_overrides)

    # --- Composable config assembly ---
    model_name = model or cfg.get("model")
    if not model_name:
        raise ValueError("No model specified. Use --model or set 'model' in config.")
    if not method:
        raise ValueError("No method specified. Use --method.")

    model_profile = _resolve_model(model_name)
    method_defaults = _resolve_method(method)

    # Build model section (with optional inline overrides from the run config)
    cfg["model"] = _build_model_section(model_profile, method)
    inline_model_overrides = cfg.pop("model_overrides", None)
    if isinstance(inline_model_overrides, dict):
        cfg["model"].update(inline_model_overrides)

    # Build method section
    method_overrides = cfg.get(method)  # e.g. inline gepa: or openevolve: overrides
    cfg[method] = _build_method_section(method, method_defaults, model_profile,
                                         overrides=method_overrides)

    # Propagate eval_num_threads to top level for OpenEvolve + ACE
    if method in ("openevolve", "ace", "ace-minimal"):
        cfg.setdefault("eval_num_threads", model_profile.get("eval_num_threads", 8))

    # Build tasks section
    task_names = tasks or cfg.get("tasks")
    single_task = task or cfg.get("task")

    if single_task:
        # Single-task mode: build 'dataset' section
        task_cfg = _resolve_task(single_task)
        # Allow inline dataset overrides
        inline_ds = cfg.get("dataset", {})
        if isinstance(inline_ds, dict):
            task_cfg.update(inline_ds)
        cfg["dataset"] = task_cfg
        cfg["task_name"] = single_task
    elif task_names:
        # Multi-task mode: build 'tasks' list
        if isinstance(task_names[0], str):
            # List of task name strings → resolve each
            resolved = []
            for name in task_names:
                task_cfg = _resolve_task(name)
                # Allow per-task inline overrides in the config
                resolved.append({"name": name, "dataset": task_cfg})
            cfg["tasks"] = resolved
        # else: already a list of {name, dataset} dicts (traditional format in composable file)

    # Auto-generate output_dir if not set
    if "output_dir" not in cfg:
        if single_task:
            cfg["output_dir"] = f"results/{method}_{single_task}"
        else:
            cfg["output_dir"] = f"results/{method}_run"

    # Append timestamp + short ID to make output_dir unique
    cfg["output_dir"] = _unique_output_dir(cfg["output_dir"])

    # Apply CLI split overrides (--train-n / --val-n / --eval-n)
    _apply_split_overrides(cfg, split_overrides)

    return cfg
