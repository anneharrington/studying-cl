#!/usr/bin/env bash
# Unified entry point: run any method on any task family with one consistent API.
#
#   ./run.sh <method> <task_family> [extra backend args...]
#
#   methods:      gepa ace            (prompt-based)
#                 sft sdft grpo sdpo  (weight updates)
#                 cartridges in_place_ttt  (compression)
#   task_family:  domain_shift | temporal_drift | discrete_updates
#
# Examples:
#   ./run.sh gepa domain_shift
#   ./run.sh sdpo temporal_drift
#   ./run.sh cartridges discrete_updates --dry-run
#
# Each family runs in its own venv (see docs/SETUP.md). This script activates a
# conventional .venv-<name> if present, otherwise it uses whatever is active.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }
[[ $# -lt 2 ]] && usage
METHOD="$1"; FAMILY="$2"; shift 2

case "$FAMILY" in domain_shift|temporal_drift|discrete_updates) ;; *)
  echo "!! unknown task_family '$FAMILY'" >&2; usage ;; esac

# Activate a conventional per-subsystem venv if it exists (no-op if already active).
activate() {
  local v="$REPO/.venv-$1"
  [[ -f "$v/bin/activate" ]] && { # shellcheck disable=SC1091
    source "$v/bin/activate"; } || echo "[run] (no $v; using active env — see docs/SETUP.md)"
}

run_prompt() {  # gepa|ace : config-driven sequential run via scripts/run.py
  activate harness
  local cfg tasks
  case "$FAMILY" in
    domain_shift)     tasks="tooluse finqa sciknoweval_bio" ;;
    temporal_drift)   tasks="finance_yr_2015 finance_yr_2016 finance_yr_2017 finance_yr_2018 finance_yr_2019 finance_yr_2020" ;;
    discrete_updates) tasks="temporalwiki_drift_s1 temporalwiki_drift_s2 temporalwiki_drift_s3" ;;
  esac
  # cl.config.load_config composes from CLI flags + configs/{methods,tasks,models}/
  # profiles, so a single thin sequential.yaml works for every (method,family) combo.
  # If a per-(method,family) override exists at configs/runs/seq_<method>_<family>.yaml
  # it wins; otherwise fall through to the generic configs/runs/sequential.yaml.
  cfg="seq_${METHOD}_${FAMILY}"
  [[ -f "configs/runs/${cfg}.yaml" ]] || cfg="sequential"
  exec python scripts/run.py --method "$METHOD" --strategy sequential \
       --config "configs/runs/${cfg}.yaml" --tasks $tasks "$@"
}

run_weight() {  # sft|sdft|grpo|sdpo : verl continual driver, ordering per family
  activate verl
  # Make the patched verl engine importable (`python -m verl.trainer.main_ppo`
  # inside run_sequential.py) — README claims this is auto-set; previously was not.
  export PYTHONPATH="$REPO/engine:${PYTHONPATH:-}"
  local ord
  case "$FAMILY" in
    domain_shift) ord=A ;; temporal_drift) ord=F ;; discrete_updates) ord=T ;;
  esac
  exec python experiments/continual/run_sequential.py \
       --method "$METHOD" --ordering "$ord" "$@"
}

run_compression() {  # cartridges|in_place_ttt : per-family run_chain
  local prefix dir
  case "$METHOD" in cartridges) prefix=cartridges; activate cartridges ;;
                    in_place_ttt) prefix=ttt; activate ttt ;; esac
  case "$FAMILY" in
    domain_shift) dir="${prefix}_domainshift" ;;
    temporal_drift) dir="${prefix}_finance" ;;
    discrete_updates) dir="${prefix}_temporalwiki" ;;
  esac
  exec bash "experiments/${dir}/run_chain.sh" "$@"
}

case "$METHOD" in
  gepa|ace)               run_prompt "$@" ;;
  sft|sdft|grpo|sdpo)     run_weight "$@" ;;
  cartridges|in_place_ttt) run_compression "$@" ;;
  *) echo "!! unknown method '$METHOD'" >&2; usage ;;
esac
