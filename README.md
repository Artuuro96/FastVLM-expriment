# FastVLM fine-tuning experiment (separate from hemblip-rep)

Fine-tunes **FastVLM-1.5B** (Apple, CVPR 2025) on the same PBC peripheral
blood cell dataset used in `hemblip-rep`, for image captioning — same task
as HemBLIP, different vision-language backbone.

Hardware: RTX A4000 (16GB VRAM), 64GB RAM, i9. Comfortable for 1.5B + LoRA
without needing 4-bit quantization.

## Why WSL2

FastVLM has no training code of its own — Apple's repo (`apple/ml-fastvlm`)
explicitly says to use the [LLaVA](https://github.com/haotian-liu/LLaVA)
codebase for fine-tuning. LLaVA's training stack depends on `flash-attn`
and `deepspeed`, both of which are unreliable to build on native Windows.
Runs under **WSL2** (Ubuntu 24.04), not PowerShell.

## Status

- [x] Data conversion script (`scripts/prepare_llava_data.py`) — 6169 train
      / 1030 val / 3099 test records generated, 0 missing images.
- [x] WSL2 (Ubuntu 24.04) — was already installed on this machine.
- [x] GPU passthrough verified (`wsl -- nvidia-smi` → RTX A4000, driver
      553.09, CUDA 12.4).
- [x] Python venv at `~/fastvlm-experiment/.venv` (WSL-side), PyTorch
      2.6.0+cu124.
- [x] `apple/ml-fastvlm` cloned to `~/fastvlm-experiment/ml-fastvlm`
      (WSL-native filesystem, not `/mnt/c`, for I/O speed).
- [x] `llava` package + train extras installed (`pip install -e ".[train]"`).
      Had to upgrade `deepspeed` past the repo's pinned `0.13.1` — that
      version doesn't import under torch 2.6 (`torch.distributed.elastic`
      API it uses was removed). Fixed with `pip install -U deepspeed`
      (landed on 0.19.5).
- [x] CUDA 12.4 toolkit (nvcc) installed in WSL2, as root (`wsl -u root`,
      no sudo password available/needed that way). The full
      `cuda-toolkit-12-4` meta-package fails on Ubuntu 24.04 — it pulls in
      `nsight-systems`, which depends on `libtinfo5` (removed from 24.04's
      repos). Installed the dev subset only (`cuda-nvcc-12-4
      cuda-cudart-dev-12-4 cuda-nvtx-12-4 libcurand-dev-12-4
      libcusparse-dev-12-4 libcublas-dev-12-4 libcusolver-dev-12-4`), which
      is all `flash-attn`/`deepspeed` actually need to compile. `CUDA_HOME`
      exported in `~/.bashrc` (see `scripts/cuda_env.sh` for the exact
      lines).
- [x] FastVLM-1.5B stage3 checkpoint downloaded directly (2.9GB;
      `get_models.sh` downloads all 6 size/stage combos, skipped that and
      `wget`'d just this one) to
      `~/fastvlm-experiment/ml-fastvlm/checkpoints/llava-fastvithd_1.5b_stage3`.
- [x] `flash-attn` — abandoned. A from-source build and both prebuilt
      cxx11abiTRUE/FALSE wheels (torch2.6+cu12+cp312) all hit the same
      `undefined symbol` on the `c10::Error` constructor (a C1/C2
      constructor-variant export mismatch) at import time. Switched to
      PyTorch's native **SDPA** attention backend instead — see
      `scripts/train_sdpa.py`, a copy of `llava/train/train_mem.py` with
      `attn_implementation="sdpa"` instead of `"flash_attention_2"`. No
      compiled extension needed, works fine on Ampere+.
- [x] 5-step smoke test on a 20-record slice — LoRA adapters attached,
      loss dropped 2.62 → 2.18, full pipeline (data → model → backward →
      optimizer step) confirmed working.
- [x] Dataset copied into WSL-native `~/fastvlm-experiment/pbc_data/`
      (303MB) — the `/mnt/c` 9p mount was bottlenecking per-image reads.
      `scripts/repoint_images.py` rewrote the `data/*.json` image paths
      to point at the copy.
- [x] Full LoRA fine-tune running (`scripts/finetune_lora.sh`, 20 epochs,
      7720 steps). ~4.7s/step steady after the I/O fix → ETA ~10h. Log:
      `outputs/train.log`. Checkpoints every 500 steps in
      `~/fastvlm-experiment/outputs/fastvlm_1.5b_lora/` (WSL-native).

## Step 1 — Convert the dataset

Already done (see Status). To regenerate after editing the CSVs:

```bash
cd fastvlm-experiment/scripts
python prepare_llava_data.py --hemblip-root ../../hemblip-rep --out-dir ../data
```

Reads `pbc_attr_v1_{train,val,test}.csv` from `hemblip-rep/PBC_dataset_normal_DIB/`
and writes `data/{train,val,test}.json` in LLaVA conversation format, e.g.:

```json
[
  {
    "id": "BNE_7323",
    "image": "/mnt/c/Users/Lyapunov/Desktop/Arturo/hemblip-rep/PBC_dataset_normal_DIB/PBC_dataset_normal_DIB/neutrophil/BNE_7323.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nDescribe this blood cell in detail."},
      {"from": "gpt", "value": "This is a neutrophil with a multi-lobulated nucleus, ..."}
    ]
  }
]
```

Captions are built from the same 11 WBCAtt attributes hemblip-rep uses for
HemBLIP, so results are comparable between the two experiments. Image paths
are already in WSL2 form (`/mnt/c/Users/...`).

**Perf note**: WSL2 reading many small files across the `/mnt/c` 9p mount
(NTFS) is noticeably slower than native ext4 — with ~10k JPEGs read every
epoch, this can bottleneck training. If data loading looks slow, `cp -r`
the `PBC_dataset_normal_DIB` folder into the WSL2 filesystem and regenerate
the JSON pointing at the copy.

## Step 2 — WSL2 + CUDA toolkit

Already done (see Status) — Ubuntu 24.04 was pre-installed, GPU passthrough
confirmed, CUDA 12.4 dev toolkit installed. `~/.bashrc` now exports
`CUDA_HOME=/usr/local/cuda-12.4` (and `PATH`/`LD_LIBRARY_PATH`).

## Step 3 — FastVLM repo + checkpoint

Already done (see Status): cloned to `~/fastvlm-experiment/ml-fastvlm`,
`pip install -e ".[train]"` (+ deepspeed upgrade), stage3 1.5B checkpoint
downloaded and extracted.

## Step 4 — Fine-tune with LoRA

`scripts/finetune_lora.sh` is the verified command — flags cross-checked
against `python3 llava/train/train.py --help` on this checkout, and the
vision-tower / projector args pulled from the checkpoint's own
`config.json`:

```json
"mm_vision_tower": "mobileclip_l_1024",
"mm_projector_type": "mlp2x_gelu",
"mm_vision_select_layer": -2,
"mm_use_im_start_end": false,
"image_aspect_ratio": "pad"
```

Conversation template is `qwen_2` (confirmed from `predict.py`'s
`--conv-mode` default, matching the checkpoint's Qwen2-based
`LlavaQwen2ForCausalLM` architecture).

No `deepspeed` launcher — single GPU, so ZeRO partitioning buys nothing;
plain HF `Trainer` with `--bf16` + `--gradient_checkpointing` is simpler
and the repo ships no `zero2.json` anyway.

Run it:

```bash
wsl
cd ~/fastvlm-experiment/ml-fastvlm
bash /mnt/c/Users/Lyapunov/Desktop/Arturo/fastvlm-experiment/scripts/finetune_lora.sh
```

Hyperparameters (`lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`,
`lr=5e-5`, 20 epochs, effective batch 16) mirror
`hemblip-rep/configs/hemblip_lora.yaml` so results stay comparable to
HemBLIP. Checkpoints land in `~/fastvlm-experiment/outputs/fastvlm_1.5b_lora`
(WSL-native, not `/mnt/c`, so checkpoint writes aren't 9p-mount-bottlenecked).
