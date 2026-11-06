"""Score ranked moment predictions against matching QVHighlights annotations."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


METRICS = ("AP", "R1_05", "R1_07", "top1_IoU", "boundary_MAE_seconds")


def read_queries(path):
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["qid"]
            if key in records:
                raise ValueError(f"{path}:{line_number}: duplicate qid {key!r}")
            records[key] = row
    if not records:
        raise ValueError(f"No query records in {path}")
    return records


def score_query(annotation, prediction, compute_ap):
    qid = annotation["qid"]
    if prediction["vid"] != annotation["vid"] or prediction["query"] != annotation["query"]:
        raise ValueError(f"Video or query mismatch for qid {qid}")
    targets = np.asarray(annotation["relevant_windows"], dtype=float)
    ranked = np.asarray(prediction["pred_relevant_windows"][:10], dtype=float)
    for windows, columns in ((targets, 2), (ranked, 3)):
        if (windows.ndim != 2 or windows.shape[1] != columns or not windows.size
                or not np.isfinite(windows).all()):
            raise ValueError(f"Malformed or missing windows for qid {qid}")
    if np.any(targets[:, 1] <= targets[:, 0]) or np.any(ranked[:, 1] < ranked[:, 0]):
        raise ValueError(f"Invalid window endpoints for qid {qid}")

    gt = [{"video-id": qid, "t-start": a, "t-end": b} for a, b in targets]
    preds = [
        {"video-id": qid, "t-start": a, "t-end": b, "score": s}
        for a, b, s in ranked
    ]
    _, ap = compute_ap((qid, gt, preds))
    top = ranked[0, :2]
    left = np.maximum(top[0], targets[:, 0])
    right = np.minimum(top[1], targets[:, 1])
    intersection = np.maximum(0.0, right - left)
    lengths = targets[:, 1] - targets[:, 0]
    ious = intersection / (top[1] - top[0] + lengths - intersection)
    closest = int(np.argmax(ious))
    best_iou = ious[closest]
    return {
        "qid": qid,
        "vid": annotation["vid"],
        "median_target_duration": float(np.median(lengths)),
        "AP": float(np.mean(ap) * 100),
        "R1_05": float(best_iou >= 0.5) * 100,
        "R1_07": float(best_iou >= 0.7) * 100,
        "top1_IoU": float(best_iou * 100),
        "boundary_MAE_seconds": float(np.abs(top - targets[closest]).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--evaluator-root", type=Path, required=True,
                        help="Checkout containing the QVHighlights standalone_eval directory")
    parser.add_argument("--per-query", type=Path, help="Write per-query scores as JSONL")
    args = parser.parse_args()

    evaluator = args.evaluator_root.resolve()
    if not (evaluator / "standalone_eval/eval.py").is_file():
        parser.error("--evaluator-root must contain standalone_eval/eval.py")
    sys.path.insert(0, str(evaluator))
    from standalone_eval.eval import compute_average_precision_detection_wrapper

    gt = read_queries(args.ground_truth)
    predictions = read_queries(args.predictions)
    if gt.keys() != predictions.keys():
        raise ValueError("Prediction and annotation query IDs must match exactly")
    rows = [
        score_query(gt[q], predictions[q], compute_average_precision_detection_wrapper)
        for q in sorted(gt)
    ]
    if args.per_query:
        args.per_query.parent.mkdir(parents=True, exist_ok=True)
        with args.per_query.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    scores = {k: float(np.mean([row[k] for row in rows])) for k in METRICS}
    scores["mAP"] = scores.pop("AP")
    print(json.dumps({"queries": len(rows), **scores}, indent=2))


if __name__ == "__main__":
    main()
