"""Check QVHighlights retrieval annotations before training or evaluation."""

import argparse
import json
import math
from pathlib import Path


LABEL_FIELDS = {"relevant_windows", "relevant_clip_ids", "saliency_scores"}


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def check_labels(row):
    duration = row["duration"]
    windows = row["relevant_windows"]
    if not isinstance(windows, list) or not windows:
        raise ValueError("relevant_windows must be a nonempty list")
    for win in windows:
        if not (isinstance(win, list) and len(win) == 2 and all(number(x) for x in win)
                and 0 <= win[0] < win[1] <= duration):
            raise ValueError("a relevant window is invalid or exceeds video duration")
    clips, scores = row["relevant_clip_ids"], row["saliency_scores"]
    if not (isinstance(clips, list) and clips and isinstance(scores, list) and len(scores) == len(clips)):
        raise ValueError("each relevant clip needs a saliency score row")
    if any(type(c) != int or c < 0 or c * 2 >= duration for c in clips):
        raise ValueError("a 2-second clip index is invalid or outside the video")
    if len(set(clips)) != len(clips):
        raise ValueError("repeated relevant clip index")
    for ratings in scores:
        if not (isinstance(ratings, list) and len(ratings) == 3 and
                all(type(x) == int and 0 <= x <= 4 for x in ratings)):
            raise ValueError("each saliency row must contain three integer scores from 0 to 4")


def check_split(path, has_labels):
    qids, vids = set(), set()
    with path.open(encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                qid = row["qid"]
                if type(qid) != int or qid < 0 or qid in qids:
                    raise ValueError("expected a unique nonnegative integer qid")
                if any(not isinstance(row[k], str) or not row[k].strip() for k in ("vid", "query")):
                    raise ValueError("vid and query must be nonempty strings")
                if not number(row["duration"]) or row["duration"] <= 0:
                    raise ValueError("duration must be positive and finite")
                if has_labels or LABEL_FIELDS.intersection(row):
                    check_labels(row)
                qids.add(qid)
                vids.add(row["vid"])
            except (ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{index}: {exc}") from exc
    if not qids:
        raise ValueError(f"{path} contains no records")
    return qids, vids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "val", "test"):
        parser.add_argument("--" + name, type=Path, metavar="JSONL")
    args = parser.parse_args()
    seen = set()
    summary = {}
    for name, path in vars(args).items():
        if path is None:
            continue
        try:
            qids, vids = check_split(path, has_labels=name != "test")
        except (ValueError, OSError) as exc:
            parser.exit(1, f"{exc}\n")
        overlap = seen & qids
        if overlap:
            parser.exit(1, f"{name}: query ID {min(overlap)} also occurs in another split\n")
        seen.update(qids)
        summary[name] = {"queries": len(qids), "distinct_video_clips": len(vids)}
    if not summary:
        parser.error("provide at least one annotation split")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
