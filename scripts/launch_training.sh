#!/usr/bin/env bash
cd ~/fastvlm-experiment/ml-fastvlm
LOG=/mnt/c/Users/Lyapunov/Desktop/Arturo/fastvlm-experiment/outputs/train.log
nohup bash /mnt/c/Users/Lyapunov/Desktop/Arturo/fastvlm-experiment/scripts/finetune_lora.sh > "$LOG" 2>&1 < /dev/null &
echo "PID=$!"
disown
