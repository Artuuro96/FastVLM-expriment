import torch
import transformers
import peft
import deepspeed
import llava

print("torch", torch.__version__, torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("deepspeed", deepspeed.__version__)
print("llava", llava.__file__)
