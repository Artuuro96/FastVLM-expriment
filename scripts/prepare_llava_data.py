"""
Converts the PBC_dataset_normal_DIB attribute CSVs (from the hemblip-rep
project) into the LLaVA-style conversation JSON that FastVLM's fine-tuning
(via the LLaVA training codebase) expects.

Caption text is built from the same 11 WBCAtt morphological attributes
used by hemblip-rep/src/data/caption_templates.py, so both experiments
are trained on equivalent ground-truth descriptions.

Usage:
    python prepare_llava_data.py --hemblip-root ../hemblip-rep --out-dir ../data
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

CELL_TYPES = [
    "basophil", "eosinophil", "erythroblast", "ig",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]


def resolve_class_root(root: Path) -> Path:
    """PBC_dataset_normal_DIB's zip export nests an extra self-named folder
    (root/PBC_dataset_normal_DIB/<class>/*.jpg) — descend into it if the
    class folders aren't directly under root. Mirrors
    hemblip-rep/src/data/wbcatt_dataset.py::_resolve_class_root."""
    if any((root / c).is_dir() for c in CELL_TYPES):
        return root
    subdirs = [
        d for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "__MACOSX"
    ] if root.is_dir() else []
    if len(subdirs) == 1:
        return resolve_class_root(subdirs[0])
    return root


INSTRUCTIONS = [
    "Describe this blood cell in detail.",
    "What morphological features do you observe in this cell?",
    "Provide a detailed description of the cell shown in the image.",
    "Examine this peripheral blood smear image and describe the cell.",
]

NUCLEUS_SHAPE_PHRASES = {
    "unsegmented-band": "a band-shaped, unsegmented nucleus",
    "unsegmented-round": "a round, unsegmented nucleus",
    "unsegmented-indented": "an indented, unsegmented nucleus",
    "segmented-bilobed": "a bilobed nucleus",
    "segmented-multilobed": "a multi-lobulated nucleus",
    "irregular": "an irregular nucleus",
}

CHROMATIN_PHRASES = {
    "densely": "densely packed chromatin",
    "loosely": "open, loosely packed chromatin",
}

CYTOPLASM_AMOUNT_PHRASES = {
    "low": "abundant cytoplasm",
    "high": "scant cytoplasm",
}

CELL_SIZE_PHRASES = {"big": "large cell size", "small": "small cell size"}

CELL_SHAPE_PHRASES = {
    "round": "round overall cell shape",
    "irregular": "irregular overall cell shape",
}


def caption_from_attributes(attrs: Dict[str, str]) -> str:
    """Mirrors hemblip-rep/src/data/caption_templates.py::caption_from_attributes,
    kept as a standalone copy so this experiment doesn't depend on importing
    across repos."""
    cell_type = attrs["cell_type"]
    nucleus = NUCLEUS_SHAPE_PHRASES.get(attrs["nucleus_shape"], f"a {attrs['nucleus_shape']} nucleus")
    chromatin = CHROMATIN_PHRASES.get(attrs["chromatin_density"], f"{attrs['chromatin_density']} chromatin")
    cytoplasm_amount = CYTOPLASM_AMOUNT_PHRASES.get(attrs["nuclear_cytoplasmic_ratio"], "cytoplasm")
    cell_size = CELL_SIZE_PHRASES.get(attrs["cell_size"], attrs["cell_size"])
    cell_shape = CELL_SHAPE_PHRASES.get(attrs["cell_shape"], attrs["cell_shape"])

    sentences = [
        f"This is a {cell_type} with {nucleus}, {chromatin}, and {cytoplasm_amount}.",
        f"The cell shows {cell_size}, {cell_shape}, and {attrs['cytoplasm_colour']} "
        f"cytoplasm with a {attrs['cytoplasm_texture']} texture"
        + (", containing vacuoles" if attrs["cytoplasm_vacuole"] == "yes" else "")
        + ".",
    ]

    if attrs["granularity"] == "yes" and attrs["granule_type"] not in ("nil", ""):
        sentences.append(f"Cytoplasmic granules are {attrs['granule_type']} and {attrs['granule_colour']}.")
    else:
        sentences.append("No prominent cytoplasmic granules are visible.")

    return " ".join(sentences)


def to_wsl_path(path: Path) -> str:
    """Convert a native Windows path to its WSL2 /mnt/<drive>/... form, since
    training runs inside WSL2 while this script runs on native Windows.
    No-op (just forward slashes) on non-Windows drives."""
    resolved = path.resolve()
    drive = resolved.drive  # e.g. "C:"
    if not drive:
        return resolved.as_posix()
    rest = resolved.as_posix()[len(drive):]  # strip "C:", keep leading "/"
    return f"/mnt/{drive[0].lower()}{rest}"


def load_split(hemblip_root: Path, split: str) -> List[Dict]:
    csv_path = hemblip_root / "PBC_dataset_normal_DIB" / f"pbc_attr_v1_{split}.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print(f"  [skip] {csv_path} missing or empty")
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_records(rows: List[Dict], hemblip_root: Path, rng: random.Random) -> List[Dict]:
    class_root = resolve_class_root(hemblip_root / "PBC_dataset_normal_DIB")
    records = []
    missing = 0
    for row in rows:
        cell_type = row["label"].strip().lower()
        image_path = class_root / cell_type / row["img_name"]
        if not image_path.exists():
            missing += 1
            continue

        attrs = {"cell_type": row["label"].strip().lower()}
        for col in (
            "cell_size", "cell_shape", "nucleus_shape", "nuclear_cytoplasmic_ratio",
            "chromatin_density", "cytoplasm_vacuole", "cytoplasm_texture",
            "cytoplasm_colour", "granule_type", "granule_colour", "granularity",
        ):
            attrs[col] = row[col]

        caption = caption_from_attributes(attrs)
        instruction = rng.choice(INSTRUCTIONS)

        records.append({
            "id": Path(row["img_name"]).stem,
            "image": to_wsl_path(image_path),
            "conversations": [
                {"from": "human", "value": f"<image>\n{instruction}"},
                {"from": "gpt", "value": caption},
            ],
        })

    if missing:
        print(f"  [warn] {missing} rows referenced missing image files (skipped)")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hemblip-root", type=Path, default=Path("../hemblip-rep"))
    parser.add_argument("--out-dir", type=Path, default=Path("../data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hemblip_root = args.hemblip_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for split in ("train", "val", "test"):
        rows = load_split(hemblip_root, split)
        records = build_records(rows, hemblip_root, rng)
        out_path = out_dir / f"{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"  [{split}] {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
