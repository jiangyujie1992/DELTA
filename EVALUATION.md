# Evaluating saved moment predictions

The two scripts in `evaluation/` score predictions and compare matched training seeds on QVHighlights. They run on CPU. Install NumPy and scikit-learn and use a checkout of the [QVHighlights evaluator](https://github.com/jayleicn/moment_detr) containing `standalone_eval/eval.py`. Install the evaluator's dependencies for your environment.

## Score one checkpoint

Use the validation annotations and a JSONL prediction file in the benchmark format. Both inputs must contain one record per query and matching `qid`, `vid`, and `query` values. Ground truth contains `relevant_windows` as start/end times in seconds. Predictions contain ranked `pred_relevant_windows` as start/end/confidence triples. Provide a nonempty ranked list for every query, highest confidence first.

```sh
python evaluation/score_predictions.py --evaluator-root /path/to/moment_detr --ground-truth /path/to/highlight_val_release.jsonl --predictions /path/to/predictions.jsonl --per-query scores/method_seed1.jsonl
```

The script computes per-query AP on the first ten predictions at IoU thresholds 0.50 to 0.95 in steps of 0.05 using the benchmark evaluator. It reports mAP, R1@0.5, R1@0.7 and mean top-1 IoU on a 0–100 scale. Boundary MAE is the mean absolute start/end error in seconds against the annotated window with maximum IoU to the first prediction. These metrics describe moment retrieval on the full evaluation query set.

## Compare paired runs

Generate per-query files for both variants of each seed, then pass one method/control pair per seed. The output JSON contains differences in percentage points, except for boundary MAE in seconds.

```sh
python evaluation/compare_predictions.py --pair scores/method_seed1.jsonl scores/control_seed1.jsonl --pair scores/method_seed2.jsonl scores/control_seed2.jsonl --duration-bins
```

The comparison averages within-query differences over matched seeds. The default 95% percentile intervals resample entire clip IDs 5,000 times, weighting each query equally within each draw. Use `--group-by source` for a sensitivity check that groups related clips by the original video ID. Intervals characterize uncertainty for fixed predictions. Training and checkpoint-selection variation are outside their scope.

The optional duration analysis groups queries by median annotated window duration (short ≤10 s, medium >10 to 30 s, long >30 s). It resamples original video IDs within each group. Each group's weighted contribution equals its fraction of queries multiplied by its mean AP difference.
