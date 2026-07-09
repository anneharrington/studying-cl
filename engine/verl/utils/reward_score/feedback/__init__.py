from verl.utils.reward_score.feedback import math
from verl.utils.reward_score.feedback import code
from verl.utils.reward_score.feedback import gpqa
from verl.utils.reward_score.feedback import mcq
from verl.utils.reward_score.feedback import tooluse
from verl.utils.reward_score.feedback import temporalwiki
from verl.utils.reward_score.feedback import finance


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    if data_source in ["code", "livecodebench", "humanevalplus"]:
        results = code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    elif data_source == "finqa":
        # FinQA golds are 4-5 sig-fig decimal ratios; exact-match and symbolic equality both
        # reject any rounded answer, which produces a near-stochastic reward and un-learnable
        # advantages under GRPO/SDPO/SDFT. Match the FinQA paper's ±0.5% rel / 1e-3 abs tolerance.
        results = math.compute_score(solution_str, ground_truth, extra_info, numeric_tolerance=True)
    elif data_source in ["math", "math500", "dapo_math", "gsm8k"]:
        results = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["gpqa"]:
        results = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval"]:
        results = mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["tooluse"]:
        results = tooluse.compute_score(solution_str, ground_truth)
    elif data_source.startswith("temporalwiki_"):
        # Per-slice data sources (e.g. temporalwiki_drift_s1, temporalwiki_stable) all use the
        # same F1 scorer; the slice tag is the metric-axis label, not a behavioral switch.
        # extra_info passes through so twiki-easy rows can opt into continuous-F1 acc
        # via extra_info.continuous_reward; original rows omit the flag (binary path).
        results = temporalwiki.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source.startswith("finance_yr_"):
        # Per-year data sources (e.g. finance_yr_2015..finance_yr_2020) all use the same
        # exact-match up/down scorer; the year tag is the metric-axis label.
        results = finance.compute_score(
            solution_str, ground_truth, extra_info=extra_info, data_source=data_source,
        )
    else:
        raise ValueError(f"Reward style {data_source} not found.")
    return results
