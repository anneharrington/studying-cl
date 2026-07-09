"""Per-example prediction logger.

Writes a pretty-printed JSON array to {output_dir}/predictions.jsonl, with
one entry per (stage, task, example) tuple — same human-readable format ACE
uses for its val_results.json. Append-only: each call seeks past the closing
"]" and rewrites it with the new entry plus a new closing bracket, so the
file is always valid JSON between calls.

Thread-safe via a process-wide lock (thread pools inside a runner share it).

Enable via config:
    logging:
      predictions_log: true

A stage is typically "baseline", "optimized", or "after_<task_name>".

Schema:
    [
      {
        "stage": "after_hotpotqa",
        "task": "ifeval",
        "index": 17,
        "llm_response": "<raw LLM output>",
        "extracted": "<parsed answer>",
        "gold": "<ground truth>",
        "metric": 0.75
      },
      ...
    ]

`question` and `general_instructions` are no longer logged — they bloat the
file and add I/O per call with no information that can't be reconstructed
from task/index + results.json's playbook_history / instructions fields.
"""

import json
import os
import threading

_lock = threading.Lock()


def render_question(task_name, ex):
    """Produce a human-readable input string for a task example.

    Mirrors each task's prompt-formatting contract (context+question for
    hotpotqa, claim for hover, prompt for ifeval, question for the rest).
    """
    if task_name == "hotpotqa":
        return f"Context: {ex.get('context', '')}\n\nQuestion: {ex.get('question', '')}"
    if task_name == "ifeval":
        return ex.get("prompt", "")
    if task_name == "hover":
        return f"Claim: {ex.get('claim', '')}"
    # sciknoweval, sciknoweval_bio, gsm8k, livebench_math, toolalpaca, tooluse, finqa
    return ex.get("question", str(ex)[:400])


def render_gold(task_name, ex):
    """Produce a human-readable gold answer for a task example."""
    if task_name == "hover":
        return ex.get("label", "")
    if task_name == "ifeval":
        ids = ex.get("instruction_id_list", [])
        return f"Must follow: {', '.join(ids)}"
    if task_name in ("tooluse", "toolalpaca"):
        return str(ex.get("golden_steps", ""))[:400]
    return str(ex.get("answer", ""))


def log_prediction(path, stage, task, question, llm_response, extracted, gold,
                   metric, index=None, general_instructions=None):
    """Append one per-example prediction record to the JSON array at `path`.

    `question` and `general_instructions` are accepted for call-site back-compat
    but are intentionally NOT written to disk — they can reach hundreds of KB
    per example with ACE playbooks or hotpot contexts, blowing up the log file
    and adding non-trivial I/O per LLM call. Reconstruct them from the
    task/index + playbook_history in results.json if needed.

    No-op if path is None.
    """
    if path is None:
        return
    entry = {
        "stage": stage,
        "task": task,
        "index": index,
        "llm_response": llm_response,
        "extracted": extracted,
        "gold": gold,
        "metric": metric,
    }
    # Pretty-print entry and indent each line 2 spaces so it nests inside the
    # top-level array, matching ACE's val_results.json style.
    entry_block = "\n".join(
        "  " + line for line in json.dumps(entry, ensure_ascii=False, indent=2).splitlines()
    )

    # Encode once; the append path operates on bytes so mid-multibyte-character
    # seeks don't raise UnicodeDecodeError (the text-mode version would crash
    # when a UTF-8 continuation byte landed at the start of our tail read).
    entry_bytes = entry_block.encode("utf-8")

    with _lock:
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                with open(path, "wb") as f:
                    f.write(b"[\n" + entry_bytes + b"\n]\n")
                return

            # Append to an existing valid "[...]" file: scan back for the last
            # "}" preceding the closing "]" and overwrite from its successor
            # with "},\n<entry>\n]\n". Read/write in binary to avoid codec
            # errors when the tail window clips a UTF-8 multibyte sequence.
            with open(path, "r+b") as f:
                size = os.fstat(f.fileno()).st_size
                tail_len = min(size, 64)
                f.seek(size - tail_len)
                tail = f.read()
                bracket_idx = tail.rfind(b"]")
                brace_idx = tail.rfind(b"}", 0, bracket_idx if bracket_idx >= 0 else len(tail))
                if brace_idx < 0:
                    # Empty array "[\n]" — overwrite the "]" with the first entry.
                    if bracket_idx < 0:
                        # Malformed — append raw with a fresh closing bracket.
                        f.seek(0, os.SEEK_END)
                        f.write(b",\n" + entry_bytes + b"\n]\n")
                        return
                    f.seek(size - tail_len + bracket_idx)
                    f.truncate()
                    f.write(entry_bytes + b"\n]\n")
                    return
                # Overwrite from right after the last "}" through EOF.
                f.seek(size - tail_len + brace_idx + 1)
                f.truncate()
                f.write(b",\n" + entry_bytes + b"\n]\n")
        except OSError:
            pass
