"""One-off: repoint image paths in the generated LLaVA JSON files from the
/mnt/c 9p mount to the WSL-native copy at ~/fastvlm-experiment/pbc_data/,
for faster per-image I/O during training."""

import json
from pathlib import Path

OLD_PREFIX = "/mnt/c/Users/Lyapunov/Desktop/Arturo/hemblip-rep/PBC_dataset_normal_DIB/PBC_dataset_normal_DIB/"
NEW_PREFIX = "/home/yoshua/fastvlm-experiment/pbc_data/PBC_dataset_normal_DIB/"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

for split in ("train", "val", "test"):
    path = DATA_DIR / f"{split}.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for r in records:
        if r["image"].startswith(OLD_PREFIX):
            r["image"] = NEW_PREFIX + r["image"][len(OLD_PREFIX):]
            changed += 1
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"{split}: {changed}/{len(records)} paths repointed")
