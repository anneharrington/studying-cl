# In-Place TTT on the domain-shift sequence (ToolUse → FinQA → SciKE-Bio)

Wires the In-Place TTT compression method into the **domain-shift** task family.

A phase = a task. Each phase continual-pretrains (TTT fast-weight updates) starting
from the previous phase's checkpoint, converts DCP→HF, then evaluates on every
task's val split through the shared `cl/` eval harness.

## Files
- `data_adapter.py` — converts the domain-shift corpora (built by
  `../cartridges_domainshift/data_adapter.py`) into VeOmni `content_split` JSONL.
- `run_chain.sh` — chains train → convert → eval over `TASKS`, threading the HF
  checkpoint from phase to phase.
- `configs/qwen3_longct_domainshift.yaml` — the TTT pretrain config (model + TTT layers).
- `install.sh` — sets up the ttt venv.

## Run (ttt env + GPU; see ../../docs/SETUP.md)
```bash
export CL_HOME=/your/scratch
# first build corpora via the cartridges domain-shift adapter, then:
TASKS="tooluse finqa sciknoweval_bio" bash run_chain.sh
```
This is a **template** like the finance/temporalwiki chains: it needs a GPU, the
ttt venv, and the corpora from `cartridges_domainshift/`.

## Eval
Serve each phase's converted HF checkpoint with vLLM, point a `configs/models/*.yaml`
profile (`EVAL_MODEL`) at it, and `scripts/eval_prompt.py` scores it on each task
with the same `cl.evals.<task>` metrics every other method reports against.
