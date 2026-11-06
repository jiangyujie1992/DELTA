"""Paired, query-weighted cluster-bootstrap contrasts for fixed predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
from score_predictions import METRICS, read_queries


def bootstrap(diffs, keys, replicates, seed):
    clusters, groups, counts = np.unique(keys, return_inverse=True, return_counts=True)
    sums = np.zeros((len(clusters), diffs.shape[1]), dtype=float)
    np.add.at(sums, groups, diffs)
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, diffs.shape[1]))
    for start in range(0, replicates, 64):
        batch = min(64, replicates - start)
        draws = rng.integers(0, len(clusters), size=(batch, len(clusters)))
        boot[start:start + batch] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    intervals = np.quantile(boot, [0.025, 0.975], axis=0)
    return {
        "queries": len(diffs),
        "clusters": len(clusters),
        "metrics": {name: {"difference": float(diffs[:, j].mean()), "ci95": intervals[:, j].tolist()}
                    for j, name in enumerate(METRICS)}
    }


def source_id(vid):
    source, start, end = vid.rsplit("_", 2)
    if len(source) != 11 or not float(end) > float(start):
        raise ValueError(f"Expected a QVHighlights source_start_end ID, got {vid}")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", required=True, type=Path,
                        metavar=("METHOD_JSONL", "CONTROL_JSONL"),
                        help="Per-query files for one seed; repeat for additional seeds")
    parser.add_argument("--group-by", choices=("clip", "source"), default="clip")
    parser.add_argument("--duration-bins", action="store_true",
                        help="Also compute duration-specific source-cluster intervals")
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17092026)
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("--replicates must be at least 2")
    reference = read_queries(args.pair[0][0])
    qids = sorted(reference)
    vids = [reference[q]["vid"] for q in qids]
    durations = np.array([reference[q]["median_target_duration"] for q in qids])
    diffs = []
    for method, control in args.pair:
        scores = []
        for path in (method, control):
            rows = read_queries(path)
            if rows.keys() != reference.keys() or any(
                rows[q]["vid"] != reference[q]["vid"] or
                rows[q]["median_target_duration"] != reference[q]["median_target_duration"]
                for q in qids
            ):
                raise ValueError(f"Query alignment failed for {path}")
            matrix = np.array([[rows[q][m] for m in METRICS] for q in qids], dtype=float)
            if not np.isfinite(matrix).all():
                raise ValueError(f"Non-finite metric in {path}")
            scores.append(matrix)
        diffs.append(scores[0] - scores[1])
    mean_diff = np.mean(diffs, axis=0)
    clusters = vids if args.group_by == "clip" else [source_id(v) for v in vids]
    report = {"seed_pairs": len(diffs), "replicates": args.replicates, "seed": args.seed,
              "group_by": args.group_by, "overall": bootstrap(mean_diff, clusters, args.replicates, args.seed)}
    if args.duration_bins:
        sources = np.array([source_id(v) for v in vids])
        bins = np.digitize(durations, [10, 30], right=True)
        report["duration_bins"] = {}
        for b, name in enumerate(("short", "medium", "long")):
            mask = bins == b
            if mask.any():
                group = bootstrap(mean_diff[mask], sources[mask], args.replicates, args.seed)
                group["weighted_mAP_contribution"] = float(mask.mean() * group["metrics"]["AP"]["difference"])
                report["duration_bins"][name] = group
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
