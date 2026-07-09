"""Track token usage and API call counts across OpenAI-compatible clients.

Works with both direct OpenAI client usage (evaluators, sequential scripts)
and OpenEvolve's internal LLM calls via monkey-patching.

Usage — wrapping a single client:

    tracker = TokenTracker()
    client = OpenAI(base_url=..., api_key=...)
    client = tracker.wrap_client(client)
    # All calls to client.chat.completions.create() are now tracked.

Usage — patching all new OpenAI clients (catches OpenEvolve internals too):

    tracker = TokenTracker()
    with tracker.patch_openai():
        result = openevolve.run_evolution(...)
    print(tracker.get_usage())

Usage — file-based tracking (works across processes, e.g. OpenEvolve workers):

    tracker = TokenTracker()
    with tracker.track_to_file():
        result = openevolve.run_evolution(...)
    print(tracker.get_usage())  # includes file-based counts

    # In evaluator files, call record_usage_to_file(response) after each API call.

Usage — DSPy/GEPA (extract from LM history):

    import dspy
    usage = get_dspy_usage()  # reads from dspy.settings.lm.history
"""

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

# Environment variable used to communicate the usage file path to worker processes
USAGE_FILE_ENV = "OPENEVOLVE_USAGE_FILE"


def usage_diff(after, before):
    """Compute the difference between two usage dicts (after - before)."""
    return {
        "prompt_tokens": after.get("prompt_tokens", 0) - before.get("prompt_tokens", 0),
        "completion_tokens": after.get("completion_tokens", 0) - before.get("completion_tokens", 0),
        "total_tokens": after.get("total_tokens", 0) - before.get("total_tokens", 0),
        "api_calls": after.get("api_calls", 0) - before.get("api_calls", 0),
    }


def get_dspy_usage(*lms):
    """Extract token usage from DSPy LM history.

    DSPy's track_usage() context manager skips cached responses, so it often
    returns empty results. This function reads directly from lm.history which
    records usage for every call (cached or not).

    Args:
        *lms: Optional dspy.LM instances to read history from. If none
              provided, reads from dspy.settings.lm (the active task LM).

    Returns:
        dict with prompt_tokens, completion_tokens, total_tokens, api_calls.
    """
    import dspy

    if not lms:
        lms = [dspy.settings.lm]

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    api_calls = 0

    for lm in lms:
        if lm is None or not hasattr(lm, "history"):
            continue
        for entry in lm.history:
            usage = entry.get("usage", {})
            if not usage:
                continue
            prompt_tokens += usage.get("prompt_tokens", 0) or 0
            completion_tokens += usage.get("completion_tokens", 0) or 0
            total_tokens += usage.get("total_tokens", 0) or 0
            api_calls += 1

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "api_calls": api_calls,
    }


def record_usage_to_file(response):
    """Append usage from an OpenAI response to the shared usage file.

    Call this from evaluator files after each API call. The file path is read
    from the OPENEVOLVE_USAGE_FILE environment variable (set by the run script).
    Each call appends one JSON line. Thread-safe via file locking.
    """
    usage_file = os.environ.get(USAGE_FILE_ENV)
    if not usage_file:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    entry = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    try:
        with open(usage_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # Don't let tracking failures break the evaluator


def _read_usage_file(path: str) -> dict:
    """Read all entries from a JSONL usage file and sum them."""
    prompt_tokens = completion_tokens = total_tokens = api_calls = 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                prompt_tokens += entry.get("prompt_tokens", 0)
                completion_tokens += entry.get("completion_tokens", 0)
                total_tokens += entry.get("total_tokens", 0)
                api_calls += 1
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "api_calls": api_calls,
    }


class TokenTracker:
    """Accumulates token usage and API call counts from OpenAI client calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.api_calls = 0
        self._usage_file = None

    def _record(self, response):
        """Extract and accumulate usage from an OpenAI response object."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        with self._lock:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.total_tokens += getattr(usage, "total_tokens", 0) or 0
            self.api_calls += 1

    def get_usage(self):
        """Return accumulated usage as a plain dict.

        Merges in-process counts (from patch_openai/wrap_client) with
        file-based counts (from evaluator worker processes).
        """
        file_usage = _read_usage_file(self._usage_file) if self._usage_file else {}
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens + file_usage.get("prompt_tokens", 0),
                "completion_tokens": self.completion_tokens + file_usage.get("completion_tokens", 0),
                "total_tokens": self.total_tokens + file_usage.get("total_tokens", 0),
                "api_calls": self.api_calls + file_usage.get("api_calls", 0),
            }

    def wrap_client(self, client):
        """Wrap an OpenAI client so all chat.completions.create() calls are tracked.

        Returns a proxy that behaves like the original client.
        """
        return _ClientProxy(client, self)

    @contextmanager
    def patch_openai(self):
        """Context manager that monkey-patches openai.OpenAI to track all new clients.

        Every OpenAI client created inside this block (including by OpenEvolve
        internals) will have its chat.completions.create() calls tracked.
        """
        import openai

        _original_init = openai.OpenAI.__init__

        tracker = self

        def _patched_init(self_client, *args, **kwargs):
            _original_init(self_client, *args, **kwargs)
            # Replace .chat.completions with a tracked proxy
            original_completions = self_client.chat.completions
            self_client.chat.completions = _CompletionsProxy(
                original_completions, tracker
            )

        openai.OpenAI.__init__ = _patched_init
        try:
            yield self
        finally:
            openai.OpenAI.__init__ = _original_init


    @contextmanager
    def track_to_file(self):
        """Context manager that sets up file-based usage tracking for worker processes.

        Creates a temp file, sets OPENEVOLVE_USAGE_FILE env var so evaluator
        workers can write to it via record_usage_to_file(), then merges the
        counts into get_usage() on exit. Also patches openai for main-process calls.
        """
        fd, path = tempfile.mkstemp(prefix="oe_usage_", suffix=".jsonl")
        os.close(fd)
        self._usage_file = path
        old_env = os.environ.get(USAGE_FILE_ENV)
        os.environ[USAGE_FILE_ENV] = path
        try:
            with self.patch_openai():
                yield self
        finally:
            if old_env is None:
                os.environ.pop(USAGE_FILE_ENV, None)
            else:
                os.environ[USAGE_FILE_ENV] = old_env
            # File stays around for get_usage() to read; clean up in __del__ or manually


class _CompletionsProxy:
    """Proxy for client.chat.completions that intercepts create()."""

    def __init__(self, completions, tracker):
        self._completions = completions
        self._tracker = tracker

    def create(self, **kwargs):
        response = self._completions.create(**kwargs)
        self._tracker._record(response)
        return response

    def __getattr__(self, name):
        return getattr(self._completions, name)


class _ChatProxy:
    """Proxy for client.chat that returns a tracked completions object."""

    def __init__(self, chat, tracker):
        self._chat = chat
        self._tracker = tracker

    @property
    def completions(self):
        return _CompletionsProxy(self._chat.completions, self._tracker)

    def __getattr__(self, name):
        return getattr(self._chat, name)


class _ClientProxy:
    """Proxy for an OpenAI client that tracks chat.completions.create() calls."""

    def __init__(self, client, tracker):
        self._client = client
        self._tracker = tracker

    @property
    def chat(self):
        return _ChatProxy(self._client.chat, self._tracker)

    def __getattr__(self, name):
        return getattr(self._client, name)
