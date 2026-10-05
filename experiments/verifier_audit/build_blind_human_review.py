"""Package the selected EventTrace audit for two independent human reviewers.

The distributed zip never includes the original AI labels or model verdicts.
The private key remains outside the zip and outside the public repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import secrets
import shutil
import zipfile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    bundle_path = args.audit_dir / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    records = bundle["records"]
    if len(records) != 49 or len({str(r["episode_id"]) for r in records}) != 12:
        raise ValueError("changed selected source")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folder = args.output_dir / "eventtrace_blind_pilot_49"
    if folder.exists():
        raise FileExistsError(folder)
    image_dir = folder / "images"
    image_dir.mkdir(parents=True)
    token = secrets.token_hex(16)
    rng = random.SystemRandom()
    order = list(range(len(records)))
    rng.shuffle(order)
    public, private = [], []
    for ordinal, index in enumerate(order, 1):
        row = records[index]
        key = f"H{ordinal:03d}_{hashlib.sha256((token + str(index)).encode()).hexdigest()[:8]}"
        paths = {}
        for view in ("before", "after"):
            meta = row["images"][view]
            source = args.audit_dir / meta["path"]
            if sha(source) != meta["sha256"]:
                raise ValueError(f"source image digest mismatch: {source}")
            name = f"{key}_{view}.jpg"
            shutil.copyfile(source, image_dir / name)
            paths[view] = "images/" + name
        item = {
            "blind_id": key,
            "instruction": row["instruction"],
            "event_source_text": row["event"]["source_text"],
            "event_type": row["event"]["type"],
            "target": row["event"].get("target"),
            "actions": row["actions"],
            "displacement_m": row["displacement"],
            "vertical_delta_m": row["vertical_delta"],
            "before_image": paths["before"],
            "after_image": paths["after"],
        }
        public.append(item)
        private.append({
            "blind_id": key,
            "episode_id": str(row["episode_id"]),
            "event_index": row["event_index"],
            "turn": row["turn"],
            "source_before_sha256": row["images"]["before"]["sha256"],
            "source_after_sha256": row["images"]["after"]["sha256"],
        })
    (folder / "items.json").write_text(json.dumps(public, indent=2) + "\n")
    with (folder / "blank_labels.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("reviewer_id", "blind_id", "label_Y_N_U",
                         "confidence_1_3", "brief_evidence"))
        for item in public:
            writer.writerow(("", item["blind_id"], "", "", ""))
    (folder / "README.md").write_text(
        "# EventTrace independent human review pilot\n\n"
        "This package contains 49 selected, transition-enriched examples "
        "from 12 R2R val-unseen episodes. It is a rubric and agreement pilot, "
        "not a population sample or a standalone verifier accuracy estimate.\n\n"
        "Two reviewers should each make their own copy, fill it independently, "
        "and return their CSVs without discussing examples. A third reviewer "
        "should adjudicate disagreements after both are locked. Do not consult "
        "model predictions, previous labels, episode paths, simulator distances, "
        "or the public audit bundle.\n\n"
        "For each event, use the full instruction, event text, actions, and "
        "before/after views. Label **Y** only if the requested transition is "
        "clearly completed; **N** only if it is clearly not completed; **U** "
        "if the available evidence cannot decide. For stop events, distinguish "
        "the stop action from visual proximity to the target. If the event "
        "requires earlier unseen route history, use U unless these views and "
        "actions settle it. Add a short evidence note and confidence 1–3.\n\n"
        "Open `index.html` locally to inspect the randomized items. Record "
        "answers in `blank_labels.csv` or use the browser's Export CSV button. "
        "Use a different reviewer_id in each independently completed file.\n"
    )
    cards = []
    for item in public:
        bid = html.escape(item["blind_id"])
        def val(k: str) -> str:
            return html.escape(str(item[k]))
        cards.append(
            f'<article><h2>{bid}</h2><p><b>Instruction:</b> {val("instruction")}</p>'
            f'<p><b>Event:</b> {val("event_source_text")} '
            f'({val("event_type")})</p><p><b>Actions:</b> '
            f'{html.escape(", ".join(item["actions"]))}; '
            f'displacement {val("displacement_m")} m, vertical '
            f'{val("vertical_delta_m")} m</p><div class="views">'
            f'<figure><img src="{item["before_image"]}"><figcaption>Before</figcaption></figure>'
            f'<figure><img src="{item["after_image"]}"><figcaption>After</figcaption></figure>'
            f'</div><div class="answer">'
            + " ".join(f'<label><input type="radio" name="{bid}" value="{code}">{code}</label>'
                       for code in "YNU")
            + f' <input class="confidence" id="c_{bid}" type="number" min="1" max="3" placeholder="confidence 1–3">'
            + f'<input class="evidence" id="e_{bid}" placeholder="brief evidence"></div></article>'
        )
    ids = json.dumps([x["blind_id"] for x in public])
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>EventTrace blind human review</title>
<style>body{font:16px system-ui;max-width:1250px;margin:auto;padding:24px;background:#f6f6f5;color:#161616}article{background:white;border:1px solid #ccc;border-radius:8px;padding:18px;margin:20px 0}.views{display:flex;gap:12px}.views figure{margin:0;width:50%}.views img{width:100%;height:auto}.answer{margin-top:15px;display:flex;gap:10px;align-items:center}.evidence{flex:1;padding:8px}.confidence{width:145px;padding:8px}button{padding:10px 16px}</style>
<h1>EventTrace independent human review pilot</h1><p>49 selected examples; Y = clearly completed, N = clearly not completed, U = insufficient evidence. Two reviewers should work independently. See README.md before labeling.</p><label>Reviewer ID <input id="reviewer" placeholder="reviewer code"></label> <button onclick="exportCsv()">Export CSV</button>
""" + "\n".join(cards) + "\n<script>const ids=" + ids + """;
function csvCell(x){return '"'+String(x??'').replaceAll('"','""')+'"'}
function exportCsv(){const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('Enter a reviewer ID');return}const rows=[['reviewer_id','blind_id','label_Y_N_U','confidence_1_3','brief_evidence']];for(const id of ids){const label=document.querySelector('input[name="'+id+'"]:checked')?.value??'';rows.push([reviewer,id,label,document.getElementById('c_'+id).value,document.getElementById('e_'+id).value])}const body=rows.map(r=>r.map(csvCell).join(',')).join('\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{type:'text/csv'}));a.download='eventtrace_review_'+reviewer+'.csv';a.click();URL.revokeObjectURL(a.href)}
</script></html>"""
    (folder / "index.html").write_text(page)
    zip_path = args.output_dir / "eventtrace_blind_pilot_49.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(folder))
    key_path = args.output_dir / "eventtrace_blind_pilot_49_PRIVATE_KEY.json"
    key_path.write_text(json.dumps({
        "schema": "eventtrace_blind_human_review_key_v1",
        "source_bundle_sha256": sha(bundle_path),
        "package_sha256": sha(zip_path),
        "selection": bundle["selection"],
        "source_episode_count": 12,
        "records": private,
    }, indent=2) + "\n")
    print(json.dumps({"package": str(zip_path), "sha256": sha(zip_path),
                      "private_key": str(key_path), "items": len(public)}))


if __name__ == "__main__":
    main()
