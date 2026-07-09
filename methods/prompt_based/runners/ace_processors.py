"""DataProcessor adapters for the ACE reference implementation.

ACE expects samples as {context, question, target} dicts and calls back into
a DataProcessor for per-task scoring. Each adapter below converts one of our
raw task examples into ACE's schema and implements the two scoring hooks
`answer_is_correct(pred, truth)` and `evaluate_accuracy(preds, truths)`.

Continuous metrics (hotpot F1, ifeval followed-fraction) are thresholded to
booleans for `answer_is_correct`; `evaluate_accuracy` reports the mean of the
underlying continuous score for parity with our other runners.

For ifeval the per-sample checker needs more than a single ground-truth
string (instruction IDs, kwargs, key, prompt), so we pack that metadata as a
JSON blob in the `target` field and unpack it inside `answer_is_correct`.

Grader unification: each processor routes through the same per-task extractor
that the benchmark harness uses in methods/prompt_based/runners/openevolve_runner.py, so an
answer counted correct during ACE training is the same one counted correct by
eval_all_tasks post-training. For ifeval and tooluse the model is instructed to
place the complete user-visible response in the JSON final_answer field so the
harness graders (which operate on the response body) can score it directly.
"""

import json

CORRECTNESS_THRESHOLD = 0.5


class HotpotqaProcessor:
    task_name = "hotpotqa"

    def process_task_data(self, raw_data):
        return [
            {
                "context": ex["context"],
                "question": ex["question"],
                "target": ex["answer"],
                "others": {"task": "hotpotqa"},
            }
            for ex in raw_data
        ]

    def _f1(self, predicted, ground_truth):
        from cl.evals.hotpot_evaluate_v1 import f1_score
        from methods.prompt_based.runners.openevolve_runner import _extract_answer
        extracted = _extract_answer(predicted or "")
        f1, _, _ = f1_score(extracted, ground_truth)
        return f1

    def answer_is_correct(self, predicted, ground_truth):
        return self._f1(predicted, ground_truth) >= CORRECTNESS_THRESHOLD

    def evaluate_accuracy(self, predictions, ground_truths):
        if not predictions:
            return 0.0
        return sum(self._f1(p, t) for p, t in zip(predictions, ground_truths)) / len(predictions)


class IFEvalProcessor:
    task_name = "ifeval"

    def process_task_data(self, raw_data):
        processed = []
        for ex in raw_data:
            target_blob = json.dumps({
                "key": ex["key"],
                "prompt": ex["prompt"],
                "instruction_id_list": ex["instruction_id_list"],
                "kwargs": ex["kwargs"],
            })
            processed.append({
                "context": "",
                "question": (
                    f"{ex['prompt']}\n\n"
                    'Your "final_answer" JSON field must contain the COMPLETE response '
                    'text that will be shown to the user, exactly as written, including '
                    'every section, formatting marker, paragraph, and required element. '
                    'Do not summarize, abbreviate, or describe the response in '
                    'final_answer — put the full response there verbatim.'
                ),
                "target": target_blob,
                "others": {"task": "ifeval"},
            })
        return processed

    def _followed_fraction(self, predicted, target_blob):
        from cl.evals.ifeval_lib.evaluation_lib import (
            InputExample,
            test_instruction_following_strict,
        )
        meta = json.loads(target_blob)
        inp = InputExample(
            key=meta["key"],
            instruction_id_list=meta["instruction_id_list"],
            prompt=meta["prompt"],
            kwargs=meta["kwargs"],
        )
        output = test_instruction_following_strict(inp, {meta["prompt"]: predicted or ""})
        n_total = len(output.follow_instruction_list)
        n_followed = sum(output.follow_instruction_list)
        return n_followed / n_total if n_total > 0 else 0.0

    def answer_is_correct(self, predicted, ground_truth):
        return self._followed_fraction(predicted, ground_truth) >= CORRECTNESS_THRESHOLD

    def evaluate_accuracy(self, predictions, ground_truths):
        if not predictions:
            return 0.0
        return sum(self._followed_fraction(p, t) for p, t in zip(predictions, ground_truths)) / len(predictions)


class HoverProcessor:
    task_name = "hover"

    def process_task_data(self, raw_data):
        return [
            {
                "context": "",
                "question": (
                    f"Claim: {ex['claim']}\n\n"
                    'Determine whether the claim is factually correct. '
                    'Your final answer must be exactly "SUPPORTED" or "NOT_SUPPORTED".'
                ),
                "target": ex["label"],
                "others": {"task": "hover"},
            }
            for ex in raw_data
        ]

    def _extract_label(self, text):
        from methods.prompt_based.runners.openevolve_runner import _extract_label as harness_extract
        return harness_extract(text or "")

    def answer_is_correct(self, predicted, ground_truth):
        return self._extract_label(predicted) == ground_truth

    def evaluate_accuracy(self, predictions, ground_truths):
        if not predictions:
            return 0.0
        correct = sum(1 for p, t in zip(predictions, ground_truths)
                      if self._extract_label(p) == t)
        return correct / len(predictions)


class SciKnowEvalBioProcessor:
    task_name = "sciknoweval_bio"

    def process_task_data(self, raw_data):
        return [
            {
                "context": "",
                "question": ex["question"],
                "target": ex["answer"].strip().upper(),
                "others": {"task": "sciknoweval_bio"},
            }
            for ex in raw_data
        ]

    def _predict(self, predicted):
        from cl.evals.sciknoweval_bio import _extract_mcq_answer as _extract_mcq_bio
        from methods.prompt_based.runners.openevolve_runner import _extract_answer
        return _extract_mcq_bio(_extract_answer(predicted or ""))

    def answer_is_correct(self, predicted, ground_truth):
        return self._predict(predicted) == ground_truth

    def evaluate_accuracy(self, predictions, ground_truths):
        if not predictions:
            return 0.0
        correct = sum(1 for p, t in zip(predictions, ground_truths)
                      if self._predict(p) == t)
        return correct / len(predictions)


class FinQAProcessor:
    task_name = "finqa"

    def process_task_data(self, raw_data):
        return [
            {
                "context": "",
                "question": ex["question"],
                "target": ex["answer"],
                "others": {"task": "finqa"},
            }
            for ex in raw_data
        ]

    def _is_match(self, predicted, ground_truth):
        # _extract_predicted_number now unwraps ACE-shaped JSON final_answer
        # before running its Answer:/last-number regex, so this matches the
        # harness path in openevolve_runner._score_and_extract verbatim.
        from cl.evals.finqa import _extract_predicted_number, _numbers_match
        pred = _extract_predicted_number(predicted or "")
        return _numbers_match(pred, ground_truth)

    def answer_is_correct(self, predicted, ground_truth):
        return self._is_match(predicted, ground_truth)

    def evaluate_accuracy(self, predictions, ground_truths):
        if not predictions:
            return 0.0
        correct = sum(1 for p, t in zip(predictions, ground_truths)
                      if self._is_match(p, t))
        return correct / len(predictions)


class LiveBenchMathProcessor:
    task_name = "livebench_math"

    def process_task_data(self, raw_data):
        # answer_type ('mcq' or three-digit) is needed at scoring time, so pack
        # both fields into target (mirrors IFEvalProcessor's JSON-blob trick).
        return [
            {
                "context": "",
                "question": ex["question"],
                "target": json.dumps({
                    "answer": ex["answer"],
                    "answer_type": ex.get("answer_type", ""),
                }),
                "others": {"task": "livebench_math"},
            }
            for ex in raw_data
        ]

    def _is_match(self, predicted, target_blob):
        from cl.evals.livebench_math import _extract_three_digit_answer, _extract_mcq_answer
        meta = json.loads(target_blob)
        gold = meta["answer"].strip()
        if meta.get("answer_type") == "mcq":
            pred = _extract_mcq_answer(predicted or "")
            return pred == gold.upper()
        pred = _extract_three_digit_answer(predicted or "")
        return pred == gold.zfill(3)

    def answer_is_correct(self, predicted, ground_truth):
        return self._is_match(predicted, ground_truth)

    def evaluate_accuracy(self, predictions, ground_truths):
        if not predictions:
            return 0.0
        correct = sum(1 for p, t in zip(predictions, ground_truths)
                      if self._is_match(p, t))
        return correct / len(predictions)


class ToolUseProcessor:
    task_name = "tooluse"

    def process_task_data(self, raw_data):
        # golden_steps is a list/dict; pack as JSON so target stays a string.
        return [
            {
                "context": "",
                "question": (
                    f"{ex['question']}\n\n"
                    'Your "final_answer" JSON field must contain the complete sequence '
                    'of Action:/Action_Input: lines exactly as specified above, with a '
                    'blank line between successive calls. Do not put the actions anywhere '
                    'else — the entire tool-call sequence must live verbatim inside '
                    'final_answer.'
                ),
                "target": json.dumps(ex["golden_steps"]),
                "others": {"task": "tooluse"},
            }
            for ex in raw_data
        ]

    def _score(self, predicted, target_blob):
        # _parse_actions now unwraps ACE-shaped JSON final_answer before
        # running its Action/Action_Input regex, matching the harness path.
        from cl.evals.toolalpaca import _parse_actions, _score_actions
        gold_steps = json.loads(target_blob)
        pred_actions = _parse_actions(predicted or "")
        score, _ = _score_actions(pred_actions, gold_steps)
        return score

    def answer_is_correct(self, predicted, ground_truth):
        return self._score(predicted, ground_truth) >= CORRECTNESS_THRESHOLD

    def evaluate_accuracy(self, predictions, ground_truths):
        if not predictions:
            return 0.0
        return sum(self._score(p, t) for p, t in zip(predictions, ground_truths)) / len(predictions)


class FinanceYrProcessor:
    """ACE adapter for the SDPO finance-bundle yearly sentiment task.

    The loader produces rows with `filing_text` (10-K excerpt) + `answer`
    ("up" or "down"). We map filing_text → ACE's `question` slot, with a
    one-line append directing the generator to emit `up` or `down`. The
    grader unwraps ACE's JSON `final_answer` shape before running the same
    `_parse_label` regex used by the cross-task harness, so train-time and
    eval-time scoring agree.
    """
    task_name = "finance_yr"

    def process_task_data(self, raw_data):
        return [
            {
                "context": "",
                "question": (
                    f"{ex['filing_text']}\n\n"
                    "Return one token: up or down."
                ),
                "target": ex["answer"],
                "others": {"task": f"finance_yr_{ex.get('year', '')}",
                           "year": ex.get("year")},
            }
            for ex in raw_data
        ]

    def _is_match(self, predicted, ground_truth):
        from cl.evals.finance_yr import _parse_label
        from methods.prompt_based.runners.openevolve_runner import _ace_final_answer
        candidate = _ace_final_answer(predicted or "") or (predicted or "")
        pred = _parse_label(candidate)
        gold = (ground_truth or "").strip().lower()
        return pred == gold and pred in ("up", "down")

    def answer_is_correct(self, predicted, ground_truth):
        return self._is_match(predicted, ground_truth)

    def evaluate_accuracy(self, predictions, ground_truths):
        if not predictions:
            return 0.0
        correct = sum(1 for p, t in zip(predictions, ground_truths)
                      if self._is_match(p, t))
        return correct / len(predictions)


class TemporalWikiProcessor:
    """ACE adapter for the TemporalWiki drift Q&A task.

    Loader rows carry `question` (= "<subject> <relation>") and `answer` (the
    object string). Pass `question` straight through to ACE's `question`
    slot. Grader: F1 >= 0.5 between extracted prediction and gold, mirroring
    `verl/utils/reward_score/feedback/temporalwiki.py`. Unwraps ACE-shaped
    JSON `final_answer` first so train-time and eval-time agree on the
    binary acc value.
    """
    task_name = "temporalwiki"

    def process_task_data(self, raw_data):
        return [
            {
                "context": "",
                "question": ex["question"],
                "target": ex["answer"],
                "others": {
                    "task": "temporalwiki_drift",
                    "subject": ex.get("subject", ""),
                    "relation": ex.get("relation", ""),
                    "slice_tag": ex.get("slice_tag", ""),
                },
            }
            for ex in raw_data
        ]

    def _f1(self, predicted, ground_truth):
        from cl.evals.temporalwiki import _extract_answer, _f1
        from methods.prompt_based.runners.openevolve_runner import _ace_final_answer
        candidate = _ace_final_answer(predicted or "") or (predicted or "")
        pred = _extract_answer(candidate)
        return _f1(pred, ground_truth or "")

    def answer_is_correct(self, predicted, ground_truth):
        return self._f1(predicted, ground_truth) >= CORRECTNESS_THRESHOLD

    def evaluate_accuracy(self, predictions, ground_truths):
        if not predictions:
            return 0.0
        # Report mean binary acc (F1>=0.5) to match the SDPO bundle's
        # val-core/<source>/acc/mean@N headline metric.
        correct = sum(1 for p, t in zip(predictions, ground_truths)
                      if self._f1(p, t) >= CORRECTNESS_THRESHOLD)
        return correct / len(predictions)


_PROCESSORS = {
    "hotpotqa": HotpotqaProcessor,
    "ifeval": IFEvalProcessor,
    "hover": HoverProcessor,
    "sciknoweval_bio": SciKnowEvalBioProcessor,
    "finqa": FinQAProcessor,
    "livebench_math": LiveBenchMathProcessor,
    "tooluse": ToolUseProcessor,
}
# Per-year / per-slice variants share one processor each. Stable is eval-only
# (cfg `optimize: false`) — never trained on, so no processor entry needed.
for _y in (2015, 2016, 2017, 2018, 2019, 2020):
    _PROCESSORS[f"finance_yr_{_y}"] = FinanceYrProcessor
for _s in ("s1", "s2", "s3"):
    _PROCESSORS[f"temporalwiki_drift_{_s}"] = TemporalWikiProcessor


def get_processor(task_name):
    if task_name not in _PROCESSORS:
        raise KeyError(
            f"No ACE DataProcessor for task '{task_name}'. "
            f"Available: {sorted(_PROCESSORS)}"
        )
    return _PROCESSORS[task_name]()
