#!/usr/bin/env bash
set -x
cd ~/fastvlm-experiment
source .venv/bin/activate
SITE=$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')
echo "SITE=$SITE"
FA_SO=$(python3 -c 'import flash_attn_2_cuda, os; print(flash_attn_2_cuda.__file__)' 2>/dev/null || find "$SITE/.." -maxdepth 1 -name 'flash_attn_2_cuda*.so')
echo "FA_SO=$FA_SO"
echo "--- ldd (libc10/libtorch lines) ---"
ldd "$FA_SO" | grep -i 'libc10\|libtorch\|not found'
echo "--- symbol in libc10.so? ---"
nm -D --demangle "$SITE/lib/libc10.so" 2>/dev/null | grep 'c10::Error::Error'
