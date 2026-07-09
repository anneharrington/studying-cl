<div align="center">
 👋 Hi, everyone!
  <br>
  We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://seed.bytedance.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/5793e67c-79bb-4a59-811a-fcc7ed510bd4">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
  <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

# In-Place Test-Time Training

**Seamlessly Endowing LLMs with Test-Time Training Ability**

Guhao Feng\*, Shengjie Luo\*, Kai Hua, Ge Zhang, Wenhao Huang, Di He, Tianle Cai

<p align="center">
  <a href="https://openreview.net/forum?id=dTWfCLSoyl">
    <img src="https://img.shields.io/badge/ICLR%202026-Oral-b31b1b?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2604.06169">
    <img src="https://img.shields.io/badge/Paper-PDF-blue?style=for-the-badge&logo=adobeacrobatreader"></a>
  <a href="#license">
    <img src="https://img.shields.io/badge/License-Apache%202.0-green?style=for-the-badge"></a>
</p>

In-Place TTT is a drop-in test-time training method for Transformer LLMs. This repository provides the training, checkpoint conversion, inference, and evaluation stack built on VeOmni, together with recommended configs for Qwen3-8B and LLaMA-3.1-8B.

## News

[2026/03] The codebase is open-sourced.
<br>
[2026/02] In-Place TTT is accepted to ICLR 2026 as an Oral presentation.

## Table of Contents

- [In-Place Test-Time Training](#in-place-test-time-training)
  - [News](#news)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Getting Started](#getting-started)
    - [Environment Setup](#environment-setup)
    - [Data Preparation](#data-preparation)
    - [Recommended Config](#recommended-config)
    - [Training](#training)
    - [Checkpoint Conversion](#checkpoint-conversion)
    - [Evaluation](#evaluation)
  - [Features](#features)
  - [License](#license)
  - [Citation](#citation)
  - [About ByteDance Seed Team](#about-bytedance-seed-team)

## Introduction

Current large language models follow a static "train then deploy" paradigm. Once deployed, model weights are frozen and cannot adapt to new information encountered during inference. This limits long-context reasoning, where useful information arrives progressively and the model would benefit from updating itself as it reads.

**In-Place Test-Time Training (In-Place TTT)** addresses this by updating a subset of model parameters, the MLP down-projection fast weights, during inference. Unlike prior TTT approaches that require architectural side modules or external memory, In-Place TTT stays inside the standard Transformer block and remains compatible with off-the-shelf autoregressive LLMs.

The method is centered around three ideas:

1. **Architectural compatibility.** Fast weights live in the existing MLP down-projection matrix, so no extra attention heads or memory modules are introduced.
2. **LM-aligned objective.** The fast-weight update is aligned with next-token prediction instead of a generic reconstruction target.
3. **Chunk-wise update.** Long sequences are split into chunks so updates can be computed efficiently and scaled to long contexts.

![In-Place TTT Method Overview](assets/pipeline.png)

As used in this repo, the end-to-end workflow is:

1. Provide your own VeOmni-compatible processed dataset and base model assets.
2. Launch continual pretraining with VeOmni through `train.sh` and `tasks/train_torch.py`.
3. Export DCP checkpoints into HuggingFace format with `scripts/merge_dcp_to_hf.py`.
4. Run TTT-aware inference and RULER evaluation with `inference_model/`, `eval.sh`, and `eval_config/`.

The repository includes recommended training configs for Qwen3-8B and LLaMA-3.1-8B, checkpoint conversion utilities, and a full RULER evaluation pipeline via OpenCompass from 4K to 256K context lengths.

## Getting Started

### Environment Setup

**Step 1.** Install PyTorch and FlashAttention:

```bash
pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip3 install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
rm flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

**Step 2.** Install VeOmni from the validated commit:

```bash
pip3 install "veomni @ git+https://github.com/ByteDance-Seed/VeOmni.git@9b91e164bea9e17f17ed490aab5e076c2335ca25"
```

**Step 3.** Install the remaining dependencies:

```bash
pip3 install liger-kernel
pip3 install byted-wandb torchdata blobfile datasets diffusers tiktoken timm
pip3 install transformers==4.57.3
pip3 install opt_einsum einops

pip3 uninstall -y byted-wandb wandb
pip3 install byted-wandb
```

**Step 4.** Optionally verify the installed VeOmni source:

```bash
python3 - <<'PY'
import json, pathlib, veomni
p = pathlib.Path(veomni.__file__).resolve().parents[1] / "veomni-0.1.0.dist-info" / "direct_url.json"
print("veomni file:", veomni.__file__)
print("direct_url:", json.loads(p.read_text()) if p.exists() else "not found")
PY
```

### Data Preparation

This repository no longer ships data-processing scripts. Provide your own processed dataset through `data.train_path`.

The recommended configs assume:

- `data.data_type=plaintext`
- `data.datasets_type=iterable`
- `data.text_keys=content_split`

For dataset argument definitions and supported loading modes, refer to the official VeOmni docs:

- [Arguments API Reference](https://veomni.readthedocs.io/en/latest/usage/arguments.html)
- [Basic Modules / Dataset & DataLoader](https://veomni.readthedocs.io/en/latest/usage/basic_modules.html)

Example:

```bash
bash train.sh tasks/train_torch.py configs/pretrain/qwen3_longct.yaml \
  --data.train_path /path/to/your_data \
  --train.output_dir /path/to/your_output_dir
```

### Recommended Config

Below is the recommended model config pattern used in the provided Qwen and LLaMA examples.

> **Tip:** Set `ttt_target: input_embed` for from-scratch pretraining, or `ttt_target: hidden_states` for continual training.

```yaml
model:
  model_path: /path/to/your_base_model
  foundation:
    ttt_layers: [0, 6, 12, 18, 24, 30, 36]
    ttt_mode: true
    ttt_proj: true
    ttt_lr: 3
    ttt_chunk: 4096

data:
  train_path: /path/to/your_data
  train_size: 20000000000
  dataloader_type: native
  datasets_type: iterable
  data_type: plaintext
  max_seq_len: 65536
  text_keys: content_split
  drop_last: true

train:
  output_dir: /path/to/your_output_dir
  data_parallel_mode: fsdp2
  global_batch_size: 64
  micro_batch_size: 1
  optimizer: adamw
  lr: 5.0e-6
  lr_warmup_ratio: 0.02
  lr_decay_style: cosine
  lr_decay_ratio: 0.90
  weight_decay: 0.1
  max_grad_norm: 1.0
  max_steps: 5000
  enable_mixed_precision: true
  enable_gradient_checkpointing: true
  enable_full_shard: true
  init_device: meta
  ckpt_manager: dcp
  save_steps: 500
  save_hf_weights: true
  use_wandb: true
```

The corresponding recommended config files are:

- `configs/pretrain/qwen3_longct.yaml`
- `configs/pretrain/llama3_longct.yaml`

### Training

Quick smoke run:

```bash
bash train.sh tasks/train_torch.py configs/pretrain/qwen3_longct.yaml \
  --train.output_dir /path/to/your_output_dir \
  --train.max_steps 1 \
  --train.use_wandb false
```

Recommended Qwen config override:

```bash
bash train.sh tasks/train_torch.py configs/pretrain/qwen3_longct.yaml \
  --train.wandb_project your_wandb_project \
  --train.wandb_name your_run_name \
  --train.output_dir /path/to/your_output_dir \
  --model.foundation '{"ttt_layers":[0,6,12,18,24,30,36],"ttt_mode":true,"ttt_proj":true,"ttt_lr":3,"ttt_chunk":4096}'
```

Recommended LLaMA config override:

```bash
bash train.sh tasks/train_torch.py configs/pretrain/llama3_longct.yaml \
  --train.wandb_project your_wandb_project \
  --train.wandb_name your_run_name \
  --train.output_dir /path/to/your_output_dir \
  --model.foundation '{"ttt_layers":[0,6,12,18,24,30,36],"ttt_mode":true,"ttt_proj":true,"ttt_lr":3,"ttt_chunk":4096}'
```

### Checkpoint Conversion

Convert VeOmni DCP checkpoints into HuggingFace format:

```bash
python scripts/merge_dcp_to_hf.py \
  --load-dir /path/to/your_checkpoint_dir

python scripts/merge_dcp_to_hf.py \
  --load-dir /path/to/your_checkpoint_dir \
  --save-dir /path/to/your_hf_checkpoint_dir \
  --model-assets-dir /path/to/your_base_model \
  --shard-size 5000000000
```

### Evaluation

Run the default RULER evaluation sweep:

```bash
bash eval.sh
```

Single-config smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -c \
  "import inference_model; from opencompass.cli.main import main; import sys; sys.argv=['opencompass','eval_config/ruler_4k.py','--debug']; main()"
```

To evaluate your own checkpoints, update `eval_config/models.py` with your model name and HuggingFace checkpoint path.

## Features

- **Drop-in TTT for standard Transformers.** In-Place TTT updates the MLP down-projection fast weights without introducing extra architectural side modules.
- **LM-aligned fast-weight updates.** The optimization target is derived for autoregressive language modeling instead of a generic reconstruction objective.
- **Long-context continual pretraining stack.** The repo includes recommended Qwen3-8B and LLaMA-3.1-8B configs built on VeOmni and FSDP2.
- **Checkpoint export path.** `scripts/merge_dcp_to_hf.py` converts VeOmni DCP checkpoints into HuggingFace format.
- **TTT-aware inference and evaluation.** `inference_model/`, `eval.sh`, and `eval_config/` cover inference and RULER evaluation through OpenCompass.
- **Long-context coverage.** The evaluation setup spans 4K, 8K, 16K, 32K, 64K, 128K, and includes a 256K config.

## License

This project is licensed under the [Apache License 2.0](./LICENSE).

## Citation

If you find this work useful for your research and applications, feel free to give us a star or cite us using:

```bibtex
@inproceedings{feng2026inplace,
  title     = {In-Place Test-Time Training},
  author    = {Feng, Guhao and Luo, Shengjie and Hua, Kai and Zhang, Ge and Huang, Wenhao and He, Di and Cai, Tianle},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {Oral Presentation},
  url       = {https://arxiv.org/abs/2604.06169}
}
```

## Contact

If you have any questions about this project, please feel free to contact us:

> Guhao Feng (fenguhao@stu.pku.edu.cn)  
> Shengjie Luo (shengjieluo@bytedance.com)

## About [ByteDance Seed Team](https://seed.bytedance.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.
