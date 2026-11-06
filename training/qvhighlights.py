"""Run native QVHighlights training on a DELTA-enabled SRF or QD-DETR checkout."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--backbone", choices=["srf", "qd_detr"], required=True)
    parser.add_argument(
        "--variant",
        choices=["baseline", "delta", "static", "qoff", "shifted"],
        default="delta",
    )
    parser.add_argument("--seed", type=int, default=2017)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--video", type=Path, nargs=2, required=True, metavar=("SLOWFAST", "CLIP")
    )
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    host = args.host.resolve(strict=True)
    if not (host / args.backbone / "delta.py").is_file():
        parser.error("Apply the host patch and copy delta.py into the host first.")
    if args.backbone == "srf" and args.captions is None:
        parser.error("SRF requires --captions.")
    if args.backbone == "qd_detr" and args.captions is not None:
        parser.error("QD-DETR uses only video and text features.")
    cmd = [
        sys.executable,
        "-m",
        args.backbone + ".train",
        "--dset_name",
        "hl",
        "--ctx_mode",
        "video_tef",
        "--exp_id",
        args.variant,
        "--seed",
        str(args.seed),
        "--train_path",
        str(args.train.resolve(strict=True)),
        "--eval_path",
        str(args.validation.resolve(strict=True)),
        "--eval_split_name",
        "val",
        "--v_feat_dirs",
        *(str(p.resolve(strict=True)) for p in args.video),
        "--t_feat_dir",
        str(args.text.resolve(strict=True)),
        "--v_feat_dim",
        "2816",
        "--t_feat_dim",
        "512",
        "--bsz",
        "32",
        "--eval_bsz",
        "100",
        "--n_epoch",
        "200",
        "--lr",
        "0.0001",
        "--wd",
        "0.0001",
        "--label_loss_coef",
        "4",
        "--lw_saliency",
        "1.0",
        "--lr_drop",
        "100" if args.backbone == "srf" else "400",
        "--results_root",
        str(args.output.resolve()),
    ]
    if args.backbone == "srf":
        cmd += [
            "--c_feat_dir",
            str(args.captions.resolve(strict=True)),
            "--c_feat_dim",
            "512",
            "--iter_num",
            "5",
            "--VTC_loss_coef",
            "0.3",
            "--CTC_loss_coef",
            "0.5",
            "--caption_loss_coef",
            "1.5",
        ]
    subprocess.run(
        cmd, cwd=host, env=dict(os.environ, DELTA_VARIANT=args.variant), check=True
    )


if __name__ == "__main__":
    main()
