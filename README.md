# DELTA

**Query-Guided Temporal Change Modeling for Video Moment Retrieval**

DELTA uses a text query to select changes between adjacent video features. It adds these changes back to the video representation before multimodal fusion, so the retrieval host can use both scene content and local temporal changes when predicting event boundaries.

## Using this repository

1. Follow the [data preparation guide](DATA.md) to prepare or check your local inputs.
2. Follow the [training guide](TRAINING.md) to install a supported host, enable the DELTA branch, and launch a run with your own features.
3. Follow the [evaluation guide](EVALUATION.md) to score ranked temporal windows and compare paired runs.

The release contains source code and usage instructions. Obtain the [QVHighlights](https://github.com/jayleicn/moment_detr) and [Charades-STA](https://github.com/jiyanggao/TALL) resources from their respective providers. Trained weights, checkpoints, datasets, predictions, and run logs are not distributed here.
