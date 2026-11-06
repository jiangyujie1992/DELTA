"""Caption-preserving Charades-STA training using the patched SRF host."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace


class CharadesDataset:
    def __init__(self, annotation_path, feature_root, caption_root, labels):
        self.data = [
            json.loads(l)
            for l in annotation_path.read_text(encoding="utf-8").splitlines()
        ]
        self.features = feature_root
        self.captions = caption_root
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from srf.span_utils import span_xx_to_cxw

        m = self.data[i]

        def norm(x):
            return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-5)

        streams = []
        for directory in ["slowfast_features", "clip_features"]:
            x = np.load(self.features / directory / (m["vid"] + ".npz"))[
                "features"
            ].astype("float32")
            x = F.interpolate(
                torch.from_numpy(x).T[None], size=75, mode="linear", align_corners=False
            )[0].T.numpy()
            streams.append(norm(x))
        video = torch.from_numpy(np.concatenate(streams, axis=-1))
        start = torch.arange(75) / 75
        video = torch.cat([video, torch.stack([start, start + 1 / 75], -1)], -1)
        query = np.load(
            self.features / "clip_text_features" / ("qid" + str(m["qid"]) + ".npz")
        )["last_hidden_state"][:32].astype("float32")
        cap = np.load(self.captions / ("qid" + str(m["qid"]) + ".npz"))["data"].astype(
            "float32"
        )
        assert cap.ndim == 3 and cap.shape[-1] == 512 and 0 < cap.shape[1] <= 32
        # Whole token sequences selected at bin centers; never interpolate token identities.
        indices = np.minimum(
            ((np.arange(75) + 0.5) * len(cap) / 75).astype(int), len(cap) - 1
        )
        cap = norm(cap[indices])
        cap = np.pad(cap, ((0, 0), (0, 32 - cap.shape[1]), (0, 0)))
        inputs = dict(
            query_feat=torch.from_numpy(norm(query)),
            video_feat=video,
            video_caption_feat=torch.from_numpy(cap),
        )
        assert all(torch.isfinite(v).all() for v in inputs.values())
        if self.labels:
            spans = (
                torch.tensor(m["relevant_windows"], dtype=torch.float32) / m["duration"]
            )
            lo = torch.arange(75) / 75
            hi = lo + 1 / 75
            overlap = (
                torch.stack(
                    [
                        (torch.minimum(hi, w[1]) - torch.maximum(lo, w[0])).clamp_min(0)
                        for w in spans
                    ]
                )
                .max(0)
                .values
            )
            pos = torch.where(overlap > 0)[0].tolist()
            neg = torch.where(overlap == 0)[0].tolist()
            assert pos
            ps = random.choices(pos, k=2)
            # Entire-video moment: identical positive/negative indices give a constant margin term.
            ns = random.choices(neg, k=2) if neg else ps.copy()
            inputs.update(
                span_labels=span_xx_to_cxw(spans),
                pos_mask=(overlap > 0).float(),
                saliency_pos_labels=ps,
                saliency_neg_labels=ns,
                saliency_all_labels=(overlap > 0).float().numpy(),
            )
        return dict(meta=m, model_inputs=inputs)


def loss(model, criterion, inputs, targets, opt):
    from srf.loss_fun.VTCLoss import VTCLoss
    from srf.loss_fun.CTCLoss import CTC_Loss
    from srf.loss_fun.VTCLoss_caption import VTCLoss_caption

    out = model(**inputs)
    ls = criterion(out, targets)
    value = sum(
        ls[k] * criterion.weight_dict[k] for k in ls if k in criterion.weight_dict
    )
    value += opt.CTC_loss_coef * CTC_Loss()(
        out["src_vid_ed"],
        out["src_txt_ed"],
        targets["src_pos_mask"],
        inputs["src_vid_mask"],
        inputs["src_txt_mask"],
    )
    value += opt.VTC_loss_coef * VTCLoss()(out["src_txt_cls_ed"], out["src_vid_cls_ed"])
    value += opt.caption_loss_coef * VTCLoss_caption()(
        out["src_cap_ed"], out["src_vid_ed"]
    )
    return value


def prediction_rows(out, metadata):
    import torch
    from srf.span_utils import span_cxw_to_xx

    scores = out["pred_logits"].softmax(-1)[..., 0].detach().cpu()
    spans = span_cxw_to_xx(out["pred_spans"]).detach().cpu()
    rows = []
    for m, ss, sc in zip(metadata, spans, scores):
        ss = (ss * m["duration"]).clamp(0, m["duration"])
        assert torch.isfinite(ss).all() and torch.isfinite(sc).all()
        windows = torch.cat([ss, sc[:, None]], -1).tolist()
        windows.sort(key=lambda x: x[2], reverse=True)
        rows.append(
            dict(
                qid=m["qid"],
                vid=m["vid"],
                query=m["query"],
                pred_relevant_windows=windows,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["host", "train", "test", "features", "captions", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--variant", choices=["baseline", "static", "delta"], default="delta"
    )
    parser.add_argument("--seed", type=int, default=2017)
    args = parser.parse_args()
    host = args.host.resolve()
    if not (host / "srf/delta.py").exists():
        parser.error("Apply the SRF patch and copy delta.py into the host first.")
    sys.path.insert(0, str(host))
    os.environ["DELTA_VARIANT"] = args.variant

    import numpy as np
    import torch
    from srf.inference import setup_model
    from srf.start_end_dataset import prepare_batch_inputs, start_end_collate

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = json.loads(Path(__file__).with_name("srf_options.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    cfg.update(
        device=torch.device("cuda"),
        resume=None,
        seed=args.seed,
        v_feat_dim=2818,
        use_tef=True,
        use_video=True,
        eval_bsz=32,
        num_workers=0,
        pin_memory=False,
        results_dir=str(args.output),
    )
    opt = SimpleNamespace(**cfg)
    train = CharadesDataset(args.train, args.features, args.captions, labels=True)
    loader = torch.utils.data.DataLoader(
        train,
        batch_size=opt.bsz,
        shuffle=True,
        num_workers=0,
        collate_fn=start_end_collate,
    )
    model, criterion, optimizer, scheduler = setup_model(opt)
    for epoch in range(opt.n_epoch):
        model.train()
        criterion.train()
        for _, packed in loader:
            inputs, targets = prepare_batch_inputs(packed, opt.device)
            optimizer.zero_grad()
            total = loss(model, criterion, inputs, targets, opt)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), opt.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch + 1}/{opt.n_epoch}", flush=True)
    torch.save(
        {"model": model.state_dict(), "epoch": opt.n_epoch}, args.output / "final.ckpt"
    )
    test = CharadesDataset(args.test, args.features, args.captions, labels=False)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=opt.eval_bsz, collate_fn=start_end_collate
    )
    model.eval()
    with torch.no_grad(), (args.output / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for metadata, packed in test_loader:
            inputs, _ = prepare_batch_inputs(packed, opt.device)
            for row in prediction_rows(model(**inputs), metadata):
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
