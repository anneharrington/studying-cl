"""Evaluate a single prompt on a task without any optimization.

Usage:
    # From a prompt file
    python scripts/eval_prompt.py --task gsm8k --model gemini-flash-lite --prompt prompts/my_prompt.txt

    # Inline prompt
    python scripts/eval_prompt.py --task hotpotqa --model qwen-3-8b --prompt-text "Answer the question.\n\n{question}"

    # Control split sizes and seed
    python scripts/eval_prompt.py --task gsm8k --model gemini-flash-lite --prompt prompts/my_prompt.txt --n 100 --seed 123

    # Use GEPA/DSPy evaluation instead of direct LLM calls
    python scripts/eval_prompt.py --task gsm8k --model gemini-flash-lite --prompt-text "Solve step by step." --mode gepa --n 50
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Load .env from CWD, then the repo root (where OPENROUTER_API_KEY / PORTKEY_API_KEY live).
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from cl.config import _resolve_model, _resolve_task
from cl.tasks import get_task


def _call_llm(client, model, prompt_text, max_retries=3,
              system_text=None, temperature=0.7, extra_body=None,
              max_tokens=8192):
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": prompt_text})
    for attempt in range(max_retries):
        try:
            kwargs = dict(model=model, messages=messages,
                          temperature=temperature, max_tokens=max_tokens)
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            response = client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  LLM call failed: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def _disable_thinking_extra(api_base):
    """Mirror evaluator.py: dashscope/aliyuncs use enable_thinking=False, others use reasoning.enabled=False."""
    base = (api_base or "").lower()
    if "dashscope" in base or "aliyuncs" in base:
        return {"enable_thinking": False}
    return {"reasoning": {"enabled": False}}


def _load_parquet_examples(parquet_path, task_name, eval_n):
    """Load eval examples from a verl/sdpo training val parquet.

    The training repo writes rows with a `prompt` list of chat messages and a
    `reward_model.ground_truth` column. We extract the user-message text as
    "question" and ground_truth as "answer" so it matches our loader_raw shape.
    """
    if task_name != "finqa":
        raise ValueError(f"Parquet loading is wired only for FinQA; got task={task_name}")
    try:
        import pyarrow.parquet as pq  # noqa: F401
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pyarrow is required to read training parquet files. Install with: "
            "uv add pyarrow"
        ) from e
    df = pd.read_parquet(parquet_path)

    def _row_to_example(row):
        # `prompt` may be a list/ndarray of {role,content} dicts or a plain string.
        prompt_field = row.get("prompt")
        # numpy arrays of object dtype come from parquet — treat them like lists.
        if hasattr(prompt_field, "tolist") and not isinstance(prompt_field, str):
            prompt_field = prompt_field.tolist()
        if isinstance(prompt_field, (list, tuple)):
            user_msgs = [m for m in prompt_field if isinstance(m, dict) and m.get("role") == "user"]
            if user_msgs:
                question = user_msgs[-1].get("content", "")
            else:
                question = " ".join(
                    str(m.get("content", "")) if isinstance(m, dict) else str(m)
                    for m in prompt_field
                )
        elif isinstance(prompt_field, str):
            question = prompt_field
        else:
            question = str(prompt_field)

        rm = row.get("reward_model")
        if isinstance(rm, dict):
            answer = rm.get("ground_truth", "")
        else:
            # Fall back to a top-level `answer` if present (some schemas).
            answer = row.get("answer", "")
        return {"question": question, "answer": str(answer), "task_type": "numeric"}

    examples = [_row_to_example(r) for _, r in df.iterrows()]
    if eval_n and eval_n > 0:
        examples = examples[:eval_n]
    return examples


def _build_extra_body(model_profile):
    """Compose extra_body from the model profile + thinking-disabled extras."""
    merged = {}
    user_eb = model_profile.get("extra_body")
    if user_eb:
        # deep enough copy for our use (no nested mutation later)
        merged = json.loads(json.dumps(user_eb))
    thinking = model_profile.get("task_thinking",
                                 model_profile.get("thinking", True))
    if not thinking:
        merged.update(_disable_thinking_extra(model_profile.get("api_base")))
    return merged or None


def eval_openevolve_style(prompt_template, task_name, examples, client, model, num_threads,
                          system_text=None, temperature=0.7, extra_body=None,
                          max_tokens=8192):
    """Evaluate by formatting the prompt template and calling the LLM directly.

    For FinQA we run an inline scoring loop so the system role + temperature
    + extra_body flow end-to-end without perturbing the shared
    openevolve_runner._score_example path that other runners depend on.
    """
    n = len(examples)
    if n == 0:
        return 0.0

    if task_name == "finqa":
        from cl.evals.finqa import _extract_predicted_number, _numbers_match

        def _score_one(ex):
            try:
                formatted = prompt_template.format(question=ex["question"])
            except (KeyError, ValueError, IndexError):
                return 0.0, "", ""
            response = _call_llm(client, model, formatted,
                                 system_text=system_text,
                                 temperature=temperature,
                                 extra_body=extra_body,
                                 max_tokens=max_tokens)
            pred = _extract_predicted_number(response)
            ok = _numbers_match(pred, ex["answer"])
            return (1.0 if ok else 0.0), response, str(pred)

        total = 0.0
        done = 0
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(_score_one, ex) for ex in examples]
            for future in as_completed(futures):
                score, _resp, _pred = future.result()
                total += score
                done += 1
                if done % 20 == 0 or done == n:
                    print(f"  {done}/{n} examples, running score={total/done:.3f}")
        return total / n

    # Non-FinQA tasks keep the existing path (no behavior change). The new
    # system/temperature flags are FinQA-only for now.
    if system_text or temperature != 0.7 or extra_body is not None:
        print("  [warn] --system / --temperature / extra_body are only wired "
              "through for --task finqa; ignoring for this task.")

    from methods.prompt_based.runners.openevolve_runner import _score_example

    total = 0.0
    done = 0
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [
            pool.submit(_score_example, ex, task_name, prompt_template, client, model)
            for ex in examples
        ]
        for future in as_completed(futures):
            score = future.result()
            total += score
            done += 1
            if done % 20 == 0 or done == n:
                print(f"  {done}/{n} examples, running score={total/done:.3f}")

    return total / n


def eval_gepa_style(instructions, task_info, examples_dspy, num_threads):
    """Evaluate by setting DSPy signature instructions and using the metric."""
    import dspy

    program = task_info["build_program"]()
    program.predict.signature = program.predict.signature.with_instructions(instructions)

    evaluator = dspy.Evaluate(
        devset=examples_dspy,
        metric=task_info["metric"],
        num_threads=num_threads,
        display_progress=True,
    )
    return evaluator(program).get("score")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a prompt on a task (no optimization)")
    parser.add_argument("--task", required=True, help="Task name (e.g. gsm8k, hotpotqa)")
    parser.add_argument("--model", required=True, help="Model profile name (e.g. gemini-flash-lite)")
    parser.add_argument("--prompt", default=None, help="Path to prompt file")
    parser.add_argument("--prompt-text", default=None, help="Inline prompt text (use \\n for newlines)")
    parser.add_argument("--mode", default="openevolve", choices=["openevolve", "gepa"],
                        help="Eval mode: 'openevolve' (direct LLM) or 'gepa' (DSPy signature)")
    parser.add_argument("--train-n", type=int, default=200, help="Training set size (skipped, default: 200)")
    parser.add_argument("--val-n", type=int, default=100, help="Validation set size (skipped, default: 100)")
    parser.add_argument("--eval-n", type=int, default=100, help="Eval set size to score on (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data sampling")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of threads (default: from model profile)")
    parser.add_argument("--system", default=None,
                        help="Path to a system prompt file (sent as role=system)")
    parser.add_argument("--system-text", default=None,
                        help="Inline system prompt (sent as role=system)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7; use 0 for greedy)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max output tokens (default: 8192)")
    parser.add_argument("--data", default=None,
                        help="Override task data path. .parquet → load training "
                             "val parquet; otherwise treated as JSONL using the "
                             "task's loader_raw.")
    parser.add_argument("-o", "--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    if not args.prompt and not args.prompt_text:
        parser.error("Specify --prompt <file> or --prompt-text <text>")
    if args.system and args.system_text:
        parser.error("Specify only one of --system or --system-text")

    # Load prompt
    if args.prompt:
        with open(args.prompt) as f:
            prompt_text = f.read().strip()
        print(f"Prompt: {args.prompt} ({len(prompt_text)} chars)")
    else:
        prompt_text = args.prompt_text.replace("\\n", "\n")
        print(f"Prompt: inline ({len(prompt_text)} chars)")

    # Load system prompt (optional)
    system_text = None
    if args.system:
        with open(args.system) as f:
            system_text = f.read().strip()
        print(f"System: {args.system} ({len(system_text)} chars)")
    elif args.system_text:
        system_text = args.system_text.replace("\\n", "\n")
        print(f"System: inline ({len(system_text)} chars)")

    # Resolve model
    model_profile = _resolve_model(args.model)
    num_threads = args.threads or model_profile.get("eval_num_threads", 8)
    extra_body = _build_extra_body(model_profile)
    if extra_body:
        print(f"Extra body: {extra_body}")
    print(f"Temperature: {args.temperature}")

    # Resolve task and load data
    task_info = get_task(args.task)
    task_dataset = _resolve_task(args.task)

    print(f"Task: {args.task}")
    print(f"Model: {model_profile['task_lm']}")
    print(f"Split: train_n={args.train_n}, val_n={args.val_n}, eval_n={args.eval_n}, seed={args.seed}")
    print(f"Evaluating on {args.eval_n} eval examples, Threads: {num_threads}")

    start_time = time.time()

    if args.mode == "openevolve":
        from openai import OpenAI

        api_key_env = model_profile.get("api_key_env", "PORTKEY_API_KEY")
        if api_key_env not in os.environ:
            print(f"Error: {api_key_env} not set")
            sys.exit(1)

        client = OpenAI(
            base_url=model_profile.get("api_base"),
            api_key=os.environ[api_key_env],
        )
        # OpenEvolve/raw OpenAI client doesn't take litellm's "openai/" prefix
        model_name = model_profile["task_lm"]
        if model_name.startswith("openai/"):
            model_name = model_name[len("openai/"):]

        # Load raw data
        if args.data and args.data.endswith(".parquet"):
            examples = _load_parquet_examples(args.data, args.task, args.eval_n)
            print(f"Loaded {len(examples)} eval examples from parquet {args.data}\n")
        else:
            data_path = args.data or task_dataset["path"]
            splits = task_info["loader_raw"](
                path=data_path,
                train_n=args.train_n,
                val_n=args.val_n,
                seed=args.seed,
                eval_n=args.eval_n,
            )
            if len(splits) < 3:
                print("Error: eval split not returned. Check that train_n + val_n + eval_n <= dataset size.")
                sys.exit(1)
            examples = splits[2]  # eval set
            print(f"Loaded {len(examples)} eval examples from {data_path} "
                  f"(skipped {args.train_n} train + {args.val_n} val)\n")

        score = eval_openevolve_style(
            prompt_text, args.task, examples, client, model_name, num_threads,
            system_text=system_text,
            temperature=args.temperature,
            extra_body=extra_body,
            max_tokens=args.max_tokens,
        )
        score_pct = score * 100

    elif args.mode == "gepa":
        import dspy
        from methods.prompt_based.gepa_method import configure_lms

        api_key_env = model_profile.get("api_key_env", "PORTKEY_API_KEY")
        configure_lms(
            task_model=model_profile.get("reflection_lm", model_profile["task_lm"]),
            reflection_model=model_profile.get("reflection_lm", model_profile["task_lm"]),
            api_base=model_profile.get("api_base"),
            api_key_env=api_key_env,
        )

        splits = task_info["loader"](
            path=task_dataset["path"],
            train_n=args.train_n,
            val_n=args.val_n,
            seed=args.seed,
            eval_n=args.eval_n,
        )
        if len(splits) < 3:
            print("Error: eval split not returned. Check that train_n + val_n + eval_n <= dataset size.")
            sys.exit(1)
        examples_dspy = splits[2]  # eval set
        print(f"Loaded {len(examples_dspy)} eval examples (skipped {args.train_n} train + {args.val_n} val)\n")

        score_pct = eval_gepa_style(prompt_text, task_info, examples_dspy, num_threads)

    elapsed = time.time() - start_time

    print(f"\n{'='*40}")
    print(f"Task:    {args.task}")
    print(f"Model:   {model_profile['task_lm']}")
    print(f"Score:   {score_pct:.2f}")
    print(f"Eval N:  {args.eval_n}")
    print(f"Seed:    {args.seed}")
    print(f"Runtime: {elapsed:.1f}s")
    print(f"{'='*40}")

    results = {
        "task": args.task,
        "model": model_profile["task_lm"],
        "model_profile": args.model,
        "score": round(score_pct, 2),
        "train_n": args.train_n,
        "val_n": args.val_n,
        "eval_n": args.eval_n,
        "seed": args.seed,
        "mode": args.mode,
        "runtime_seconds": round(elapsed, 1),
        "prompt": prompt_text,
        "system": system_text,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "extra_body": extra_body,
        "data": args.data,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
