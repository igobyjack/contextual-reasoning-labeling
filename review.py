"""
review.py

Build an HTML review sheet for the labels produced by label.py.
Each scenario-frames image is shown next to its safe flag, hazard keywords,
and reasoning, so a human can quickly eyeball the labels before merging.

Images are downscaled to thumbnails and inlined as base64 (they never change),
but the labels themselves are fetched live from labels.jsonl by the page at
load time. That means once review.html is generated you can re-run label.py
(or hand-edit labels.jsonl) and just refresh the browser to see new labels --
no need to regenerate the HTML.

Because the page fetches labels.jsonl, it must be served over HTTP rather than
opened via file:// (browsers block fetch on file:// URLs). Serve the folder:

    python -m http.server 8000
    # then open http://localhost:8000/review.html

Usage:
    huggingface-cli login          # if the dataset is private
    python review.py               # (re)build review.html once (embeds images)
    python -m http.server 8000     # serve the folder
    # edit labels.jsonl -> refresh browser to see updates
"""

import io
import json
import base64
import html

from datasets import load_dataset

# --------------------------------------------------------------------------
# Config (kept in sync with label.py)
# --------------------------------------------------------------------------
DATASET_NAME = "podolinsky/Contextual-Reasoning"
DATA_FILES = "scenario-frames/**"
SPLIT = "train"
IMAGE_COLUMN = "image"
CKPT = "labels.jsonl"
OUT_HTML = "review.html"

THUMB_MAX = 512          # longest side of each thumbnail, in pixels


def thumb_data_uri(pil_img):
    img = pil_img.convert("RGB")
    img.thumbnail((THUMB_MAX, THUMB_MAX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def main():
    print(f"Loading {DATASET_NAME} [{DATA_FILES}] [{SPLIT}] ...")
    ds = load_dataset(DATASET_NAME, data_files=DATA_FILES, split=SPLIT)

    images = {}
    for i in range(len(ds)):
        images[i] = thumb_data_uri(ds[i][IMAGE_COLUMN])
        print(f"  thumbnailed {i + 1}/{len(ds)}", end="\r")
    print()

    images_json = json.dumps(images)
    ckpt_json = json.dumps(CKPT)
    title = html.escape(f"{DATASET_NAME} [{DATA_FILES}]")

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Label review</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f5f7; color: #1d1d1f; }}
  header {{ position: sticky; top: 0; background: #fff; padding: 16px 24px;
           border-bottom: 1px solid #ddd; box-shadow: 0 1px 4px rgba(0,0,0,.05); }}
  header h1 {{ margin: 0 0 4px; font-size: 18px; }}
  header .stats {{ font-size: 13px; color: #555; }}
  header .controls {{ margin-top: 8px; font-size: 13px; }}
  header button {{ font: inherit; padding: 4px 12px; border: 1px solid #ccc;
                  border-radius: 6px; background: #fff; cursor: pointer; }}
  header button:hover {{ background: #f0f0f0; }}
  header label {{ margin-left: 12px; color: #555; }}
  .row {{ display: flex; gap: 20px; background: #fff; margin: 12px 24px;
         padding: 12px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
  .imgcol img {{ max-width: 320px; max-height: 320px; border-radius: 6px; display: block; }}
  .txtcol {{ flex: 1; }}
  .head {{ font-weight: 600; font-size: 15px; margin-bottom: 8px; }}
  .badge {{ font-size: 12px; padding: 2px 8px; border-radius: 999px; margin-left: 8px; color: #fff; }}
  .safe {{ background: #2e9e4f; }}
  .unsafe {{ background: #d1342f; }}
  .err {{ background: #8a8a8a; }}
  .kws {{ margin-bottom: 8px; }}
  .kw {{ display: inline-block; background: #ffe9e9; color: #a11; font-size: 12px;
        padding: 2px 8px; border-radius: 6px; margin: 0 4px 4px 0; }}
  .reason {{ font-size: 14px; line-height: 1.45; }}
  .muted {{ color: #999; font-size: 12px; }}
</style></head><body>
<header>
  <h1>Label review &mdash; {title}</h1>
  <div class="stats" id="stats">loading labels&hellip;</div>
  <div class="controls">
    <button id="refresh">Refresh labels</button>
    <label><input type="checkbox" id="auto"> auto-refresh every 5s</label>
  </div>
</header>
<div id="rows"></div>
<script>
const IMAGES = {images_json};
const CKPT = {ckpt_json};

function badge(rec) {{
  const b = document.createElement("span");
  b.className = "badge";
  if (rec.error) {{ b.classList.add("err"); b.textContent = "ERROR"; }}
  else if (rec.safe === 1) {{ b.classList.add("safe"); b.textContent = "SAFE"; }}
  else if (rec.safe === 0) {{ b.classList.add("unsafe"); b.textContent = "UNSAFE"; }}
  else {{ b.classList.add("err"); b.textContent = "?"; }}
  return b;
}}

function renderRow(idx, rec) {{
  const row = document.createElement("div");
  row.className = "row";

  const imgcol = document.createElement("div");
  imgcol.className = "imgcol";
  if (IMAGES[idx]) {{
    const img = document.createElement("img");
    img.src = IMAGES[idx];
    img.loading = "lazy";
    imgcol.appendChild(img);
  }}

  const txtcol = document.createElement("div");
  txtcol.className = "txtcol";

  const head = document.createElement("div");
  head.className = "head";
  head.textContent = "#" + idx + " ";
  head.appendChild(badge(rec));

  const kws = document.createElement("div");
  kws.className = "kws";
  const list = rec.hazard_keywords || [];
  if (list.length) {{
    for (const k of list) {{
      const s = document.createElement("span");
      s.className = "kw";
      s.textContent = k;
      kws.appendChild(s);
    }}
  }} else {{
    const s = document.createElement("span");
    s.className = "muted";
    s.textContent = "none";
    kws.appendChild(s);
  }}

  const reason = document.createElement("div");
  reason.className = "reason";
  reason.textContent = rec.error ? rec.error : (rec.reasoning || "");

  txtcol.appendChild(head);
  txtcol.appendChild(kws);
  txtcol.appendChild(reason);
  row.appendChild(imgcol);
  row.appendChild(txtcol);
  return row;
}}

async function refresh() {{
  const stats = document.getElementById("stats");
  let text;
  try {{
    const resp = await fetch(CKPT + "?_=" + Date.now(), {{ cache: "no-store" }});
    if (!resp.ok) throw new Error(resp.status + " " + resp.statusText);
    text = await resp.text();
  }} catch (e) {{
    stats.textContent = "Failed to load " + CKPT + ": " + e.message +
      " (serve this folder with `python -m http.server` and open over http://).";
    return;
  }}

  const labels = {{}};
  for (const line of text.split("\\n")) {{
    const s = line.trim();
    if (!s) continue;
    try {{ const r = JSON.parse(s); labels[r.idx] = r; }} catch (e) {{}}
  }}

  const idxs = Object.keys(labels).map(Number).sort((a, b) => a - b);
  let nSafe = 0, nUnsafe = 0, nErr = 0;
  const container = document.getElementById("rows");
  container.textContent = "";
  for (const i of idxs) {{
    const rec = labels[i];
    if (rec.error) nErr++;
    else if (rec.safe === 1) nSafe++;
    else if (rec.safe === 0) nUnsafe++;
    container.appendChild(renderRow(i, rec));
  }}

  stats.innerHTML = idxs.length + " labeled &nbsp;|&nbsp; <b>" + nUnsafe +
    "</b> unsafe &nbsp;|&nbsp; <b>" + nSafe + "</b> safe &nbsp;|&nbsp; <b>" +
    nErr + "</b> errored &nbsp;|&nbsp; <span class=\\"muted\\">updated " +
    new Date().toLocaleTimeString() + "</span>";
}}

document.getElementById("refresh").addEventListener("click", refresh);
let timer = null;
document.getElementById("auto").addEventListener("change", (e) => {{
  if (e.target.checked) timer = setInterval(refresh, 5000);
  else clearInterval(timer);
}});
window.addEventListener("focus", refresh);
refresh();
</script>
</body></html>"""

    with open(OUT_HTML, "w") as f:
        f.write(doc)

    print(f"Wrote {OUT_HTML} ({len(images)} images embedded).")
    print("Labels are now loaded live from labels.jsonl.")
    print("Serve the folder and open over http://, e.g.:")
    print("    python -m http.server 8000")
    print("    # then open http://localhost:8000/review.html")
    print("After editing labels.jsonl, just refresh the browser.")


if __name__ == "__main__":
    main()
