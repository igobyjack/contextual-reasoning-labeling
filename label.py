"""
label.py

Pre-label a HuggingFace image dataset for infant/baby safety using a VLM.
Produces, per image, a concise reasoning trace + a binary safe flag (1/0) +
a list of hazard keywords drawn from a fixed vocabulary.

Output is written incrementally to labels.jsonl (checkpointed / resumable).
Run human review on labels.jsonl afterward, then merge with merge.py.

Usage:
    export PROMPT_UPSAMPLER_API_TOKEN=...
    export PROMPT_UPSAMPLER_ENDPOINT_URL=...
    huggingface-cli login          # if the dataset is private
    python label.py
"""

import os
import io
import json
import base64

from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATASET_NAME = "podolinsky/Contextual-Reasoning"
DATA_FILES = "scenario-frames/**"          # only label the scenario-frames subset
SPLIT = "train"
IMAGE_COLUMN = "image"                     
MODEL = "gpt-5.4"
CKPT = "labels.jsonl"

# --------------------------------------------------------------------------
# Hazard vocabulary (kept fixed so labels stay consistent)
# --------------------------------------------------------------------------
HAZARD_KEYWORDS = [
    # choking
    "coin", "battery", "small_toy", "marble", "button",
    # electrical
    "outlet", "cord", "wire",
    # sharp
    "knife", "scissors", "glass", "blade",
    # suffocation
    "plastic_bag", "loose_blanket", "pillow_in_crib",
    # fall
    "stairs", "elevated_surface", "unguarded_height",
    # burn
    "stove", "hot_liquid", "candle", "iron", "sunlight",
    # poison
    "cleaning_product", "medication", "toxic_plant", "alcohol",
    # water
    "pool", "bathtub", "bucket",
    # strangulation
    "blind_cord", "strap", "necklace", "dangerous_appliance"
]
HAZARD_SET = set(HAZARD_KEYWORDS)
VOCAB_STR = ", ".join(HAZARD_KEYWORDS)

# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
PROMPT = f"""Analyze this image for infant/baby safety hazards.
Return ONLY valid JSON, no other text:
{{"reasoning": "...", "safe": 0 or 1, "hazard_keywords": [list]}}

Rules:
- reasoning: ONE or TWO short sentences. Name the specific hazard(s)
  and why they endanger a baby. If safe, briefly state why nothing
  poses a risk. No preamble, no hedging.
- Use keywords ONLY from this list: [{VOCAB_STR}]
- safe = 1 only if NO hazard from the list is present
- safe = 0 if any hazard is present
- hazard_keywords is [] when safe = 1
- reasoning must be consistent with safe and hazard_keywords"""


client = OpenAI(
    api_key=os.environ["PROMPT_UPSAMPLER_API_TOKEN"],
    base_url=os.environ["PROMPT_UPSAMPLER_ENDPOINT_URL"],
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def encode_image(pil_img):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def label_image(pil_img):
    b64 = encode_image(pil_img)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    if not raw:
        raise ValueError("empty response from model")
    data = json.loads(raw)

    kws = [k for k in data.get("hazard_keywords", []) if k in HAZARD_SET]
    safe = 1 if len(kws) == 0 else 0
    reasoning = (data.get("reasoning") or "").strip()

    return {
        "safe": safe,
        "hazard_keywords": kws,
        "reasoning": reasoning,
    }


def load_checkpoint(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    # tolerate a truncated final line from an interrupted run
                    continue
                done[r["idx"]] = r
    return done


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print(f"Loading {DATASET_NAME} [{DATA_FILES}] [{SPLIT}] ...")
    ds = load_dataset(DATASET_NAME, data_files=DATA_FILES, split=SPLIT)
    print(ds)

    if IMAGE_COLUMN not in ds.column_names:
        raise ValueError(
            f"Image column '{IMAGE_COLUMN}' not found. "
            f"Available columns: {ds.column_names}"
        )

    done = load_checkpoint(CKPT)
    if done:
        print(f"Resuming: {len(done)} images already labeled.")

    errors = 0
    with open(CKPT, "a") as f:
        for i in tqdm(range(len(ds))):
            if i in done:
                continue
            try:
                result = label_image(ds[i][IMAGE_COLUMN])
            except Exception as e:
                errors += 1
                result = {
                    "safe": None,
                    "hazard_keywords": [],
                    "reasoning": "",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            rec = {"idx": i, **result}
            f.write(json.dumps(rec) + "\n")
            f.flush()

    if errors:
        print(f"Note: {errors} images errored (safe is null).")
    print(f"Done. Labels written to {CKPT}")
    print("Next: human-review the file, then run merge.py.")


if __name__ == "__main__":
    main()