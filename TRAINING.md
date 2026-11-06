# DELTA training

The training code attaches the query-guided residual to the projected video stream in SRF or QD-DETR. SRF consumes caption features as well as video and query features. QD-DETR recomputes the residual with the paired negative query for its negative-pair pass. The host handles the optimizer, losses and validation for QVHighlights. The Charades-STA driver implements the fixed-duration adaptation for SRF.

## Set up a host

The commands below use a Linux shell, a CUDA device and the Python dependencies listed by the selected upstream repository. Run the first line from the DELTA repository. Put the host checkouts and experiment outputs in a separate working directory. The patches target the revisions shown below.

```bash
DELTA="$(pwd)"
mkdir -p ../delta-hosts && cd ../delta-hosts
git clone https://github.com/VisualAIKHU/SRF.git
git -C SRF checkout 7b662944dcd1a85d6b8f0251b178277cf704cecb
git -C SRF apply --ignore-space-change "$DELTA/training/srf.patch"
cp "$DELTA/training/delta.py" SRF/srf/delta.py

git clone https://github.com/wjun0830/QD-DETR.git
git -C QD-DETR checkout f8628f79f7c651b586300b142dbe9b85e43857cc
git -C QD-DETR apply --ignore-space-change "$DELTA/training/qd_detr.patch"
cp "$DELTA/training/delta.py" QD-DETR/qd_detr/delta.py
```

Use the installation and feature download instructions from the [SRF](https://github.com/VisualAIKHU/SRF) or [QD-DETR](https://github.com/wjun0830/QD-DETR) repository for your selected host. The patches include small portions of upstream context. Their original MIT notices are in `training/LICENSE-srf.txt` and `training/LICENSE-qd_detr.txt`.

## QVHighlights

Set `QV` to a local directory containing the two benchmark annotation files and extracted feature directories shown below. The paths are explicit arguments and may point to separate volumes. Pass the SlowFast stream first and the CLIP stream second.

```bash
python "$DELTA/training/qvhighlights.py" --host SRF --backbone srf \
  --variant delta --seed 2017 \
  --train "$QV/qvhighlights_train.jsonl" --validation "$QV/qvhighlights_val.jsonl" \
  --video "$QV/slowfast_features" "$QV/clip_features" --text "$QV/clip_text_features" \
  --captions "$QV/caption_features_internvl" --output ./runs/srf-delta

python "$DELTA/training/qvhighlights.py" --host QD-DETR --backbone qd_detr \
  --variant delta --seed 2018 \
  --train "$QV/qvhighlights_train.jsonl" --validation "$QV/qvhighlights_val.jsonl" \
  --video "$QV/slowfast_features" "$QV/clip_features" --text "$QV/clip_text_features" \
  --output ./runs/qd-delta
```

Use `--variant static` for the capacity-matched content input or `--variant baseline` for the host alone. The SRF diagnostic settings are `qoff` (learned query-independent gate) and `shifted` (valid differences rotated inside each sequence). The launcher exports the variant to the native trainer. For subsequent standalone host inference, export the same `DELTA_VARIANT` used for the checkpoint, for example `export DELTA_VARIANT=delta`.

## Charades-STA

Set `CH` to the local feature root. Each video ID indexes an NPZ file with a `features` array in `slowfast_features/` and `clip_features/`. Query files in `clip_text_features/` are named `qid<ID>.npz` with a `last_hidden_state` array. The separate SRF caption directory provides `qid<ID>.npz` files with a `data` array of shape (caption clips, tokens, 512). Each annotation line needs `qid`, `vid`, `query` and `duration` in seconds; training rows also need `relevant_windows` in seconds. Match feature IDs to annotation IDs on both splits.

```bash
python "$DELTA/training/charades.py" --host SRF --variant delta --seed 2017 \
  --train "$CH/charades_sta_train.jsonl" --test "$CH/charades_sta_val.jsonl" \
  --features "$CH" --captions "$CH/caption_features_sfc" \
  --output ./runs/charades-delta
```

Pass the benchmark test annotations to `--test`; the example follows the feature mirror's filename. The adapter interpolates each visual stream to 75 bins and selects complete caption token sequences at bin centers. It trains on overlap-derived saliency targets, saves the final epoch locally and writes scored windows in seconds to `predictions.jsonl`. Choose a new output directory for each run.
