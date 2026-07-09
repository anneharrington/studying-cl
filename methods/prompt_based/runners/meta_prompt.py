"""Shared META_PROMPT configuration for multi-task optimization.

The meta prompt is guidance about multi-task retention. It now lives in
different places depending on the method:
- GEPA: appended to metric feedback strings (seen by reflection LM, not task LM)
- OpenEvolve: included in the evolution LM's system message (not task prompt)

Config:
    meta_prompt:
      enabled: true   # default true; set false to disable
      text: "..."     # optional override of default text
"""

DEFAULT_META_PROMPT = (
    "You will be evaluated and reevaluated on multiple different tasks in sequence, "
    "so try to retain information and strategies that work across different task types."
)


def get_meta_prompt(cfg):
    """Get the meta prompt from config, or None if disabled.

    Config format:
        meta_prompt:
          enabled: true/false
          text: "custom text"

    Also accepts:
        meta_prompt: false          # disable
        meta_prompt: "custom text"  # enable with custom text
        (absent)                    # enable with default
    """
    mp_cfg = cfg.get("meta_prompt")

    if mp_cfg is None:
        return DEFAULT_META_PROMPT

    if isinstance(mp_cfg, bool):
        return DEFAULT_META_PROMPT if mp_cfg else None

    if isinstance(mp_cfg, str):
        return mp_cfg

    if isinstance(mp_cfg, dict):
        if not mp_cfg.get("enabled", True):
            return None
        return mp_cfg.get("text", DEFAULT_META_PROMPT)

    return DEFAULT_META_PROMPT
