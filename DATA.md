# Preparing QVHighlights annotations

Use the [QVHighlights benchmark](https://github.com/jayleicn/moment_detr) release JSONL files with the corresponding pre-extracted features. Keep the original numeric query IDs and string video IDs: training loads query features by query ID, while evaluation joins predictions to annotations using both IDs and the query text.

Check one or more annotation files before a run. The checker uses the Python standard library.

```sh
python data_tools/check_qvh_splits.py --train data/highlight_train_release.jsonl --val data/highlight_val_release.jsonl --test data/highlight_test_release.jsonl
```

The command reports the number of queries and distinct video clips in each split. It checks the two-second clip grid, positive video durations, annotated window bounds, one row of three saliency ratings per relevant clip, and distinct query IDs within and across the supplied files. The public test split can omit all annotation fields. Invalid input exits with a nonzero status and a diagnostic. These checks concern annotations; feature availability is a separate prerequisite for training.

For saved predictions and paired confidence intervals, follow [EVALUATION.md](EVALUATION.md).
