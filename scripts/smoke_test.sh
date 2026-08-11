#!/usr/bin/env bash
# Quick end-to-end validation: 5 training steps on a tiny slice of train.json,
# before committing to the full 20-epoch run.
set -e

EXPERIMENT_ROOT="/mnt/c/Users/Lyapunov/Desktop/Arturo/fastvlm-experiment"
CHECKPOINT="./checkpoints/llava-fastvithd_1.5b_stage3"
OUTPUT_DIR="$HOME/fastvlm-experiment/outputs/smoke_test"

export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

source "$HOME/fastvlm-experiment/.venv/bin/activate"

rm -rf "$OUTPUT_DIR"

python3 "$EXPERIMENT_ROOT/scripts/train_sdpa.py" \
    --lora_enable True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --model_name_or_path "$CHECKPOINT" \
    --version qwen_2 \
    --data_path "$EXPERIMENT_ROOT/data/smoke_train.json" \
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
    --max_steps 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --save_strategy no \
    --learning_rate 5e-5 \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none
