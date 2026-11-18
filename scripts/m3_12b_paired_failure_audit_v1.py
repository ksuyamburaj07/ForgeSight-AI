"""ForgeSight M3.12B-v1: compare immutable M3.8 tiled and M3.12A image scores.

Descriptive paired failure analysis only; NO test-label-based threshold selection.
stdlib only; no image decoding, no GPU, no model inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

OLD_SHA = "ea0354d971d52ad1ef1d3adea9eb9fdd36849f19318bf266eb4e97ac32f11f72"
NEW_SHA = "0d33b11936c95241c503a318c447d6cad3105c5e0a93d2ec3e5068a8e9d4f35b"
CATS = ("pcb1", "pcb2", "pcb3", "pcb4")
OUTCOME_ORDER = (
    "defect_both_detect", "defect_new_only", "defect_tiled_only", "defect_both_miss",
    "normal_both_flagged", "normal_new_only_false_alarm", "normal_tiled_only_false_alarm", "normal_both_accepted",
)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def indexed(rows):
    result = {}
    for row in rows:
        image = row["image"]
        if image in result:
            raise ValueError(f"Duplicate image: {image}")
        if int(row["label"]) not in (0, 1):
            raise ValueError(f"Invalid label in {image}")
        result[image] = row
    return result


def outcomes(old_report: dict, new_report: dict) -> tuple[dict, list[dict]]:
    rows = []
    per_cat = {}
    for cat in CATS:
        old = old_report["categories"][cat]["tiled"]
        new = new_report["categories"][cat]
        old_threshold = float(old["frozen_threshold"])
        new_threshold = float(new["development_frozen_threshold"])
        left = indexed(old["per_image"])
        right = indexed(new["per_image"])
        if set(left) != set(right):
            raise ValueError(f"Image cohort mismatch in {cat}: old {len(left)}, new {len(right)}")
        if len(left) != (200 if cat in ("pcb1", "pcb2") else 201):
            raise ValueError(f"Incorrect test-image count in {cat}")
        counter = Counter()
        labels = Counter()
        old_cm, new_cm = Counter(), Counter()
        for image in sorted(left):
            a, b = left[image], right[image]
            if int(a["label"]) != int(b["label"]):
                raise ValueError(f"Label mismatch for {image}")
            label = int(a["label"])
            pred_old = float(a["score"]) > old_threshold
            pred_new = float(b["score"]) > new_threshold
            labels[label] += 1
            if label:
                outcome = "defect_" + ("both_detect" if pred_old and pred_new else "new_only" if pred_new else "tiled_only" if pred_old else "both_miss")
                old_cm["TP" if pred_old else "FN"] += 1
                new_cm["TP" if pred_new else "FN"] += 1
            else:
                outcome = "normal_" + ("both_flagged" if pred_old and pred_new else "new_only_false_alarm" if pred_new else "tiled_only_false_alarm" if pred_old else "both_accepted")
                old_cm["FP" if pred_old else "TN"] += 1
                new_cm["FP" if pred_new else "TN"] += 1
            counter[outcome] += 1
            rows.append({
                "category":cat,"image":image,"ground_truth_label":label,
                "outcome":outcome,
                "tiled_score":float(a["score"]),"tiled_frozen_threshold":old_threshold,
                "tiled_flag":int(pred_old),"tiled_score_minus_own_threshold":float(a["score"])-old_threshold,
                "m312a_score":float(b["score"]),"m312a_frozen_threshold":new_threshold,
                "m312a_flag":int(pred_new),"m312a_score_minus_own_threshold":float(b["score"])-new_threshold,
            })
        if labels != {0:(100 if cat in ("pcb1","pcb2") else 101),1:100}:
            raise ValueError(f"Unexpected class counts: {cat} {labels}")
        for name, computed, original in (("tiled", old_cm,old["image_metrics"]),("M3.12A",new_cm,new["image_metrics"])):
            if any(computed[k] != original[k] for k in ("TP","FP","TN","FN")):
                raise ValueError(f"{cat} {name} confusion mismatch")
        per_cat[cat] = {"outcomes":{k:counter[k] for k in OUTCOME_ORDER},
                        "tiled_confusion":dict(old_cm),"m312a_confusion":dict(new_cm),
                        "category_test_images":len(left)}
    if len(rows) != 802 or len({r["image"] for r in rows}) != 802:
        raise ValueError("Expected exactly 802 unique images")
    total = Counter()
    for row in rows:
        total[row["outcome"]] += 1
    assert sum(total.values()) == 802
    return {
        "experiment":"M3.12B-v1 immutable paired image-score failure audit",
        "status":"DESCRIPTIVE_RETROSPECTIVE_ONLY_NOT_INDEPENDENT",
        "source_sha256":{"m3_8_tiled":OLD_SHA,"m3_12a":NEW_SHA},
        "test_label_based_threshold_selection":False,
        "models_loaded":False,"images_decoded":False,"thresholds_changed":False,
        "method":"Pair image paths; compare each image score to its own model's previously frozen category threshold; categorize TP/FN/FP/TN disagreements",
        "warning":"Category case lists come from previously inspected benchmark labels; use for qualitative failure analysis only, not model/threshold selection or independent validation",
        "categories":per_cat,
        "totals":{k:total[k] for k in OUTCOME_ORDER},
        "cohort_images":len(rows),
    }, rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiled", type=Path,required=True)
    parser.add_argument("--m312a", type=Path,required=True)
    parser.add_argument("--outdir",type=Path,default=Path("reports/eda"))
    args=parser.parse_args()
    for path, expected in ((args.tiled,OLD_SHA),(args.m312a,NEW_SHA)):
        actual=sha256(path)
        if actual != expected:
            raise SystemExit(f"SOURCE HASH MISMATCH: {path}: expected {expected}; got {actual}")
    old=json.loads(args.tiled.read_text(encoding="utf8"))
    new=json.loads(args.m312a.read_text(encoding="utf8"))
    report,rows=outcomes(old,new)
    args.outdir.mkdir(parents=True,exist_ok=True)
    report_path=args.outdir/"m3_12b_paired_failure_audit_v1.json"
    csv_path=args.outdir/"m3_12b_paired_case_manifest_v1.csv"
    if report_path.exists() or csv_path.exists():
        raise SystemExit("Audit output already exists: immutable version 1 files will not be overwritten")
    with report_path.open("x",encoding="utf8") as f:
        json.dump(report,f,indent=2,sort_keys=True); f.write("\n")
    with csv_path.open("x",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print("M3.12B DESCRIPTIVE FAILURE AUDIT — not an independent test")
    for cat in CATS:
        print(cat,report["categories"][cat]["outcomes"])
    print("ALL",report["totals"])
    print(f"JSON: {report_path} SHA256: {sha256(report_path)}")
    print(f"CSV:  {csv_path} SHA256: {sha256(csv_path)}")
    print("No model inference, threshold changes, or image decoding performed.")

if __name__=="__main__":
    main()
