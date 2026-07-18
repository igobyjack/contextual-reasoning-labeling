"""
merge.py

Attach the labels produced by label.py (and human-reviewed in labels.jsonl)
back onto the scenario-frames images, then push the result to the Hub as a
new dataset with columns: image, safe, hazard_keywords, reasoning.

Usage:
    huggingface-cli login
    python merge.py <target-repo-id>
    # e.g. python merge.py your-username/Contextual-Reasoning-labeled

    # or set the repo once and just run `python merge.py`:
    export TARGET_REPO=your-username/Contextual-Reasoning-labeled
    python merge.py
"""

import os
import sys
import json

from datasets import load_dataset

# --------------------------------------------------------------------------
# Config (kept in sync with label.py)
# --------------------------------------------------------------------------
DATASET_NAME = "podolinsky/Contextual-Reasoning"
DATA_FILES = "scenario-frames/**"
SPLIT = "train"
IMAGE_COLUMN = "image"
CKPT = "labels.jsonl"

# Set to True to drop rows that errored during labeling (safe is null).
DROP_ERRORS = True


def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run label.py first to produce labels."
        )
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            labels[r["idx"]] = r
    return labels


def resolve_target_repo():
    if len(sys.argv) > 1:
        return sys.argv[1]
    env = os.environ.get("TARGET_REPO")
    if env:
        return env
    raise SystemExit(
        "No target repo given. Usage:\n"
        "    python merge.py <target-repo-id>\n"
        "or set TARGET_REPO in the environment."
    )


def main():
    target_repo = resolve_target_repo()

    print(f"Loading {DATASET_NAME} [{DATA_FILES}] [{SPLIT}] ...")
    ds = load_dataset(DATASET_NAME, data_files=DATA_FILES, split=SPLIT)

    labels = load_labels(CKPT)
    missing = [i for i in range(len(ds)) if i not in labels]
    if missing:
        raise SystemExit(
            f"{len(missing)} images have no label (idx: {missing[:10]}...). "
            f"Finish labeling before merging."
        )

    safe_col = [labels[i].get("safe") for i in range(len(ds))]
    kw_col = [labels[i].get("hazard_keywords", []) for i in range(len(ds))]
    reason_col = [labels[i].get("reasoning", "") for i in range(len(ds))]

    ds = ds.add_column("safe", safe_col)
    ds = ds.add_column("hazard_keywords", kw_col)
    ds = ds.add_column("reasoning", reason_col)

    if DROP_ERRORS:
        before = len(ds)
        ds = ds.filter(lambda r: r["safe"] is not None)
        dropped = before - len(ds)
        if dropped:
            print(f"Dropped {dropped} errored row(s) with null safe.")

    print(ds)
    print(f"Pushing {len(ds)} rows to {target_repo} ...")
    ds.push_to_hub(target_repo, split=SPLIT)
    print(f"Done. https://huggingface.co/datasets/{target_repo}")


if __name__ == "__main__":
    main()
