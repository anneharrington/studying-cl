# In-Place TTT — three parallel CL benchmarks

This dir + two siblings (`ttt_sweep`, `ttt_temporalwiki`) wire up the In-Place
Test-Time Training method (ByteDance, ICLR 2026 oral) into our 6-method CL
comparison. Same datasets and eval scoring as our other 5 methods (SFT, SDFT,
GRPO, SDPO, Cartridges) — just a different training mechanism.

Source: <https://github.com/ByteDance-Seed/In-Place-TTT>

## What In-Place TTT does

Continual-pretraining method that updates **MLP down-projection "fast weights"**
during training. Recommended Qwen3-8B placement: TTT modules at layers
`[0, 6, 12, 18, 24, 30, 36]`, `ttt_lr=3, ttt_chunk=4096, ttt_target=hidden_states`.
Outer optimizer: AdamW, `lr=5e-6`, cosine schedule, `gbs=64, mbs=1, max_seq_len=16384`.

Output is a DCP checkpoint that gets merged into a standard HuggingFace
causal-LM via their `scripts/merge_dcp_to_hf.py`. Eval then loads the HF
checkpoint and runs the same val parquets as our other methods.

## Three benchmarks (parallel dirs)

| Dir | Order | Phases | Train inputs |
|---|---|---|---|
| `scripts/ttt_finance/` | F | 6 (y2015..y2020) | per-year 10-K bodies (reuse `cartridges_finance/corpora/`) |
| `scripts/ttt_sweep/` | B | 3 (tooluse→finqa→bio) | per-task training set, flattened to plaintext |
| `scripts/ttt_temporalwiki/` | T | 3 (S1, S2, S3) | per-slice Wikipedia article-revision text |

Each dir has its own `data_adapter.py`, `run_chain.sh`, `run_chain.sbatch`,
`eval_*.py`. The venv install is shared (`install.sh` in this dir works for
all three).

## CL framing (matches our other methods)

Phase i is initialised from phase (i-1)'s converted HF checkpoint and
TTT-pretrained on phase i's data.

```
phase 1: base Qwen3-8B   --TTT pretrain on slice_1--> ckpt_1 --eval matrix
phase 2: ckpt_1          --TTT pretrain on slice_2--> ckpt_2 --eval matrix
...
```

For finance specifically, slice_i = `cumulative_y{i}/corpus.txt` (matches
cartridges_finance and the SDPO/SFT setup). For sweep, slice_i is just task_i's
training set. For temporalwiki, slice_i is the Wikipedia diff for time pair i.

## Outputs

```
/workspace/home/nayan/ttt_<benchmark>/runs/<RUN_TAG>/
├── data/<phase>/data.jsonl            # adapter output (VeOmni plaintext format)
├── train/<phase>/dcp/                 # raw DCP checkpoints
├── train/<phase>/hf/                  # converted HF checkpoints
├── logs/{train,convert,eval}_*.log
└── manifest.json

/workspace/home/nayan/results/<benchmark>_predictions/ttt/<phase>/<eval_target>.parquet
```

Where `<benchmark>` is `finance` / `sweep_orderB` / `temporalwiki` and
`<eval_target>` is year/task/slice.

## Run-tag convention (matches SDPO repo)

- finance:        `finance-cl-<TS>-<SHA4>_ttt_orderF_nothink_s42`
- sweep order B:  `sweep-<TS>-<SHA4>_ttt_orderB_nothink_s42`
- temporalwiki:   `twiki-cl-<TS>-<SHA4>_ttt_orderT_nothink_s42`

## Status (start in this order)

- [x] `install.sh` (this dir; works for all 3 benchmarks)
- [x] `data_adapter.py` (finance — converts existing corpora to JSONL)
- [x] `configs/qwen3_longct_finance.yaml` (with our paths)
- [ ] `run_chain.sh` (finance)
- [ ] `eval_ttt.py` (finance — load HF ckpt, predict up/down)
- [ ] `run_chain.sbatch` + smoke variant
- [ ] **then** sibling: `scripts/ttt_sweep/` (orderB)
- [ ] **then** sibling: `scripts/ttt_temporalwiki/` (orderT)

Currently scaffolding only. Will flesh out after the cartridges_finance prod
run completes (don't want to thrash GPU resources during that run).
