"""
Entry point mirroring llava/train/train_mem.py, but using PyTorch's native
SDPA attention backend instead of flash-attention-2.

Avoids a flash-attn C++ ABI mismatch (undefined symbol on the c10::Error
constructor — a C1/C2 constructor-variant export mismatch between the
prebuilt wheel and this torch 2.6.0+cu124 build) that persisted across
both a from-source build and both prebuilt cxx11abiTRUE/FALSE wheels.
SDPA is natively supported by transformers/PyTorch on Ampere+ GPUs and
needs no extra compiled extension.

Run from the ml-fastvlm repo root:
    python3 /mnt/c/.../fastvlm-experiment/scripts/train_sdpa.py ...
"""

from llava.train.train_qwen import train

if __name__ == "__main__":
    train(attn_implementation="sdpa")
