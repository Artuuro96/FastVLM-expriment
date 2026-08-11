import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
for split in ("train", "val", "test"):
    d = json.loads((DATA_DIR / f"{split}.json").read_text(encoding="utf-8"))
    ok = sum(1 for r in d if os.path.exists(r["image"]))
    print(f"{split}: {ok}/{len(d)} images resolve, sample={d[0]['image']}")
