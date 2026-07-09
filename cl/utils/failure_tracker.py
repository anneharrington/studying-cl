"""Track LM failures across DSPy and direct API calls.

Works generically with any LLM — captures parse failures, refusals,
timeouts, and other errors.

Usage (DSPy/GEPA):

    tracker = FailureTracker()
    tracker.start()       # attach to DSPy's error logger
    # ... run GEPA optimization and evaluation ...
    tracker.stop()
    print(tracker.summary())  # {"total": 5, "by_stage": {"after_hotpotqa": 3, ...}}

Usage (OpenEvolve / direct API):

    tracker = FailureTracker()
    # In your scoring loop:
    tracker.record("after_hotpotqa", error="LLM returned empty response")
    print(tracker.summary())

Both can be combined — DSPy errors are auto-captured, direct API errors
are recorded manually.
"""

import logging
import threading


class FailureTracker:
    """Count and categorize LM failures."""

    def __init__(self):
        self._lock = threading.Lock()
        self._failures = []
        self._current_stage = "unknown"
        self._handler = None

    def set_stage(self, stage):
        """Set the current stage label for subsequent failures."""
        self._current_stage = stage

    def record(self, stage=None, error=""):
        """Manually record a failure (for direct API calls)."""
        with self._lock:
            self._failures.append({
                "stage": stage or self._current_stage,
                "error": error[:200],  # truncate long errors
            })

    def start(self):
        """Attach a handler to DSPy's parallelizer logger to auto-capture errors."""
        self._handler = _DSPyErrorHandler(self)
        logger = logging.getLogger("dspy.utils.parallelizer")
        logger.addHandler(self._handler)

    def stop(self):
        """Detach the DSPy error handler."""
        if self._handler:
            logger = logging.getLogger("dspy.utils.parallelizer")
            logger.removeHandler(self._handler)
            self._handler = None

    @property
    def total(self):
        with self._lock:
            return len(self._failures)

    def summary(self):
        """Return a summary dict suitable for JSON serialization."""
        with self._lock:
            by_stage = {}
            for f in self._failures:
                stage = f["stage"]
                by_stage[stage] = by_stage.get(stage, 0) + 1
            return {
                "total": len(self._failures),
                "by_stage": by_stage,
            }


class _DSPyErrorHandler(logging.Handler):
    """Logging handler that counts ERROR-level messages from DSPy."""

    def __init__(self, tracker):
        super().__init__(level=logging.ERROR)
        self._tracker = tracker

    def emit(self, record):
        msg = record.getMessage()
        # Extract a short error description from DSPy's error format
        # Format: "Error for Example({...}): <error message>"
        error_short = msg
        if ": " in msg:
            # Take everything after the last ": " as the error type
            parts = msg.split(": ", 1)
            if len(parts) > 1:
                error_short = parts[1][:200]
        self._tracker.record(error=error_short)
