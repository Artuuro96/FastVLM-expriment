#!/usr/bin/env bash
# LoRA fine-tune of FastVLM-1.5B (stage3 checkpoint) on the PBC blood-cell
# captioning data. Run from inside WSL2, from the ml-fastvlm repo root:
#
#   cd ~/fastvlm-experiment/ml-fastvlm
#   bash /mnt/c/Users/Lyapunov/Desktop/Arturo/fastvlm-experiment/scripts/finetune_lora.sh
#
# Verified against `python3 llava/train/train.py --help` on this checkout
# (apple/ml-fastvlm, llava package v1.2.2.post1). Single-GPU (RTX A4000,
# 16GB) — no deepspeed launcher needed; plain HF Trainer + bf16 +
# gradient checkpointing fits comfortably for a 1.5B model with LoRA.
#
# Uses train_sdpa.py (this folder) instead of llava/train/train_mem.py:
# train_mem.py hardcodes attn_implementation="flash_attention_2", but
# flash-attn wouldn't import here (C++ ABI mismatch on the c10::Error
# constructor, reproduced with a from-source build and both prebuilt
# cxx11abiTRUE/FALSE wheels for torch2.6+cu12+cp312). SDPA is PyTorch's
# native attention backend — no extra compiled extension, works fine on
# Ampere+.
#
# Hyperparameters (lora_r/alpha/dropout, lr, epochs) mirror
# hemblip-rep/configs/hemblip_lora.yaml so results are comparable to
# HemBLIP.

set -e

EXPERIMENT_ROOT="/mnt/c/Users/Lyapunov/Desktop/Arturo/fastvlm-experiment"
CHECKPOINT="./checkpoints/llava-fastvithd_1.5b_stage3"
OUTPUT_DIR="$HOME/fastvlm-experiment/outputs/fastvlm_1.5b_lora"

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

source "$HOME/fastvlm-experiment/.venv/bin/activate"

python3 "$EXPERIMENT_ROOT/scripts/train_sdpa.py" \
    --lora_enable True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --model_name_or_path "$CHECKPOINT" \
    --version qwen_2 \
    --data_path "$EXPERIMENT_ROOT/data/train.json" \
    --image_folder / \
    --vision_tower mobileclip_l_1024 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type flat \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
