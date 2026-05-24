"""Phase I: LabelStore の擬似ラベルから model を fine-tune.

Usage:
    python scripts/phase_i_fine_tune.py \
        --component score \
        [--video-ids video_02,video_03 ... | --all] \
        [--dry-run]

対応 component:
    - score: ScoreFineTuner (digit テンプレ更新)
    - next:  NextFineTuner (CNN fine-tune)
    - chain: ChainFineTuner (VideoChainTracker 閾値 grid search)

出力:
    - score → models/ui_templates/score_digits/digit_N.png 上書き (rollback 可)
    - next  → CNN state を fine-tune (--save-to で保存)
    - chain → data/verify/chain_tracker_calibration.json (rollback 可)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.score_ocr import DEFAULT_TEMPLATE_DIR
from src.self_supervised.cell_color_fine_tuner import (
    DEFAULT_BASE_MODEL as CELL_DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_PATH as CELL_DEFAULT_OUTPUT_PATH,
    CellColorFineTuner,
)
from src.self_supervised.chain_fine_tuner import (
    DEFAULT_CALIBRATION_PATH as CHAIN_CALIBRATION_PATH,
    ChainFineTuner,
)
from src.self_supervised.label_store import LabelStore
from src.self_supervised.next_fine_tuner import NextFineTuner
from src.self_supervised.pseudo_label import (
    COMPONENT_CELL,
    COMPONENT_CHAIN,
    COMPONENT_NEXT,
    COMPONENT_SCORE,
    PseudoLabelSample,
)
from src.self_supervised.score_fine_tuner import ScoreFineTuner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True,
                         choices=["score", "next", "chain", "cell_color"])
    parser.add_argument("--video-ids", type=str, default="",
                         help="カンマ区切り video_id list (空 + --all で全部)")
    parser.add_argument("--all", action="store_true",
                         help="ストア内の全 video_id を対象")
    parser.add_argument("--store-root", type=Path,
                         default=Path("data/pseudo_labels"))
    parser.add_argument("--template-dir", type=Path,
                         default=Path(DEFAULT_TEMPLATE_DIR),
                         help="score のみ。テンプレ出力先")
    parser.add_argument("--cnn-model", type=Path,
                         default=Path("models/cnn_global_best.pt"),
                         help="next のみ。fine-tune 対象 CNN")
    parser.add_argument("--save-to", type=Path,
                         default=Path("models/cnn_pseudo_finetuned.pt"),
                         help="next のみ。fine-tune 後の保存先")
    parser.add_argument("--epochs", type=int, default=3,
                         help="next のみ")
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="next のみ")
    parser.add_argument("--chain-calibration-path", type=Path,
                         default=CHAIN_CALIBRATION_PATH,
                         help="chain のみ。calibration JSON 出力先")
    parser.add_argument("--chain-min-confidence", type=float,
                         default=0.0,
                         help="chain のみ。サンプル confidence の下限")
    parser.add_argument("--cell-base-model", type=Path,
                         default=Path(CELL_DEFAULT_BASE_MODEL),
                         help="cell_color のみ。base CNN model path")
    parser.add_argument("--cell-save-to", type=Path,
                         default=Path(CELL_DEFAULT_OUTPUT_PATH),
                         help="cell_color のみ。fine-tune 後の保存先")
    parser.add_argument("--cell-arch", type=str, default="small",
                         choices=["small", "large"],
                         help="cell_color のみ。case D: 'large' で CnnPatchClassifierLarge (100KB)")
    parser.add_argument("--augment", action="store_true",
                         help="cell_color のみ。B-2 色順序対称 augment を有効化")
    parser.add_argument("--enable-topo-filter", action="store_true",
                         help="cell_color のみ。S-7 TopoFilter noise 除去を有効化")
    parser.add_argument("--topo-n-clusters", type=int, default=8,
                         help="cell_color のみ。S-7 cluster 数 (有効化時)")
    parser.add_argument("--topo-min-agreement", type=float, default=0.6,
                         help="cell_color のみ。S-7 多数決最小合意率")
    parser.add_argument("--topo-max-samples", type=int, default=200000,
                         help="cell_color のみ。S-7 OOM ガード上限 (0=無制限)")
    parser.add_argument("--topo-disable-minibatch", action="store_true",
                         help="cell_color のみ。S-7 で MiniBatchKMeans 無効化")
    parser.add_argument("--class-balance", action="store_true",
                         help="cell_color のみ。class weight で mode collapse 緩和")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                         help="cell_color のみ。focal loss gamma (0=off)")
    parser.add_argument("--logit-adjust-tau", type=float, default=0.0,
                         help="cell_color のみ。logit adjustment tau (0=off)")
    parser.add_argument("--oversample-alpha", type=float, default=0.0,
                         help="cell_color のみ。CReST 風 minority oversample 強度 (0=off, 1=均等)")
    parser.add_argument("--freeze-ojama-logit", action="store_true",
                         help="cell_color のみ。cycle 57: 最終 Linear 層の OJAMA "
                              "class row の gradient を 0 にして ojama 認識を保護")
    parser.add_argument("--dry-run", action="store_true",
                         help="集計のみ、更新しない")
    parser.add_argument(
        "--apply-review-filter", action="store_true",
        help="cycle 32c (2026-05-19): seed_review_filter.json で定義された "
             "video × color の除外を適用 (= ユーザー目視で NG 確定済 seed を学習対象外)",
    )
    parser.add_argument(
        "--review-filter-path", type=Path,
        default=Path("data/verify/seed_review_filter.json"),
        help="--apply-review-filter 時に参照する JSON path",
    )
    parser.add_argument(
        "--use-circle-mask", action="store_true",
        help="cycle 32g: 学習時 patch に円形マスク (= 四隅 0 塗り) を適用。 "
             "推論時も同じく USE_CIRCLE_MASK=True を設定する必要あり",
    )
    return parser.parse_args()


def _load_review_filter(path: Path) -> dict[str, set[int]]:
    """seed_review_filter.json を読んで video_id → exclude_colors set を返す."""
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    filters = data.get("filters", {})
    out: dict[str, set[int]] = {}
    for vid, spec in filters.items():
        colors = spec.get("exclude_colors", [])
        if colors:
            out[vid] = set(int(c) for c in colors)
    return out


def _resolve_video_ids(args: argparse.Namespace) -> list[str]:
    if args.all:
        return LabelStore.list_videos(root=args.store_root)
    if not args.video_ids:
        raise ValueError("--video-ids または --all を指定")
    return [v.strip() for v in args.video_ids.split(",") if v.strip()]


def _load_samples(
    component: str, video_ids: list[str], root: Path,
    skip_video_color: dict[str, set[int]] | None = None,
) -> list[PseudoLabelSample]:
    """video_id 横断で擬似ラベル sample を load.

    cycle 32c (2026-05-19): skip_video_color で video × color 単位の除外を
    指定可能。 ユーザー目視で全 NG 確定した seed を学習対象外にする。
    """
    skip = skip_video_color or {}
    samples: list[PseudoLabelSample] = []
    n_skipped = 0
    for vid in video_ids:
        store = LabelStore(video_id=vid, root=root)
        bad_colors = skip.get(vid, set())
        for s in store.load(component):
            if bad_colors and int(s.label) in bad_colors:
                n_skipped += 1
                continue
            samples.append(s)
    if n_skipped > 0:
        print(f"[phase_i] review-filter skipped {n_skipped} samples")
    return samples


def _summary(samples: list[PseudoLabelSample]) -> dict:
    """ラベル分布 + 平均 confidence を簡易表示."""
    out: dict = {"n_total": len(samples), "by_confidence": {}, "by_label": {}}
    for s in samples:
        bucket = round(s.confidence, 2)
        out["by_confidence"][bucket] = out["by_confidence"].get(bucket, 0) + 1
        key = str(s.label)
        out["by_label"][key] = out["by_label"].get(key, 0) + 1
    return out


def _fine_tune_score(
    samples: list[PseudoLabelSample], args: argparse.Namespace,
) -> dict:
    tuner = ScoreFineTuner(template_dir=args.template_dir)
    if args.dry_run:
        return {"dry_run": True, "n_samples": len(samples)}
    return tuner.fine_tune(samples)


def _fine_tune_chain(
    samples: list[PseudoLabelSample], args: argparse.Namespace,
) -> dict:
    tuner = ChainFineTuner(
        calibration_path=args.chain_calibration_path,
        min_confidence=args.chain_min_confidence,
    )
    if args.dry_run:
        return {"dry_run": True, "n_samples": len(samples)}
    return tuner.fine_tune(samples)


def _fine_tune_cell_color(
    samples: list[PseudoLabelSample], args: argparse.Namespace,
) -> dict:
    """cell 色 CNN の自己教師あり fine-tune.

    cycle 71v (案 D, 2026-05-13): --cell-arch large で CnnPatchClassifierLarge
    (= 100KB 中規模 CNN) を使用.
    cycle 56 (= 2026-05-21): --cell-base-model が指定され、 かつファイル存在する場合は
    state_dict を引き継ぐ (= 真の fine-tune)。 cycle 55 で「scratch 化」 した
    教訓に対応。 base model 未指定 / ファイル不在なら従来通り scratch 訓練。
    """
    topo_max = args.topo_max_samples if args.topo_max_samples > 0 else None
    cnn = None
    if getattr(args, "cell_arch", "small") == "large":
        from src.patch_classifier import CnnPatchClassifierLarge
        cnn = CnnPatchClassifierLarge()
        # base model から state_dict 引き継ぎ (cycle 56〜)
        base_path = getattr(args, "cell_base_model", None)
        if base_path is not None and Path(base_path).is_file():
            import torch
            state = torch.load(
                str(base_path), map_location="cpu", weights_only=True,
            )
            cnn._model.load_state_dict(state)
            print(f"[phase_i] loaded Large base model from {base_path}")
        # GPU 切替試行
        try:
            import os as _os
            import torch as _torch
            if (
                _os.environ.get("CUDA_VISIBLE_DEVICES", "all") != ""
                and _torch.cuda.is_available()
            ):
                cnn.to_device("cuda")
        except Exception:
            pass
    tuner = CellColorFineTuner(
        base_model_path=args.cell_base_model,
        output_path=args.cell_save_to,
        lr=args.lr,
        epochs=args.epochs,
        cnn=cnn,
        augment=args.augment,
        enable_topo_filter=args.enable_topo_filter,
        topo_n_clusters=args.topo_n_clusters,
        topo_min_agreement=args.topo_min_agreement,
        topo_max_samples=topo_max,
        topo_use_minibatch=not args.topo_disable_minibatch,
        class_balance=args.class_balance,
        focal_gamma=args.focal_gamma,
        logit_adjustment_tau=args.logit_adjust_tau,
        oversample_alpha=args.oversample_alpha,
        freeze_ojama_logit=args.freeze_ojama_logit,
    )
    if args.dry_run:
        return {"dry_run": True, "n_samples": len(samples)}
    return tuner.fine_tune(samples)


def _fine_tune_next(
    samples: list[PseudoLabelSample], args: argparse.Namespace,
) -> dict:
    from src.patch_classifier import CnnPatchClassifier
    cnn = CnnPatchClassifier()
    if args.cnn_model.is_file():
        import torch
        state = torch.load(
            str(args.cnn_model), map_location="cpu", weights_only=True,
        )
        cnn._model.load_state_dict(state)
    # GPU 切替
    try:
        import torch as _t
        if _t.cuda.is_available():
            cnn.to_device("cuda")
    except Exception:
        pass
    tuner = NextFineTuner(cnn=cnn, lr=args.lr, epochs=args.epochs)
    if args.dry_run:
        return {"dry_run": True, "n_samples": len(samples)}
    metrics = tuner.fine_tune(samples)
    # 保存
    args.save_to.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(cnn._model.state_dict(), str(args.save_to))
    metrics["saved_to"] = str(args.save_to)
    return metrics


def main() -> None:
    args = parse_args()
    # cycle 32g: 円形マスク global flag を学習開始前に設定
    if getattr(args, "use_circle_mask", False):
        from src.patch_classifier import set_circle_mask_enabled
        set_circle_mask_enabled(True)
        print("[phase_i] cycle 32g: circle mask ENABLED (= patch 四隅 0 塗り)")
    video_ids = _resolve_video_ids(args)
    print(f"[phase_i] component={args.component} videos={video_ids}")
    component_map = {
        "score": COMPONENT_SCORE,
        "next": COMPONENT_NEXT,
        "chain": COMPONENT_CHAIN,
        "cell_color": COMPONENT_CELL,
    }
    component = component_map[args.component]
    skip_video_color: dict[str, set[int]] | None = None
    if args.apply_review_filter:
        skip_video_color = _load_review_filter(args.review_filter_path)
        if skip_video_color:
            print(
                f"[phase_i] review-filter loaded: "
                f"{len(skip_video_color)} videos × colors to exclude",
            )
        else:
            print(
                f"[phase_i] review-filter requested but no rules at "
                f"{args.review_filter_path}",
            )
    samples = _load_samples(
        component, video_ids, args.store_root,
        skip_video_color=skip_video_color,
    )
    print(f"[phase_i] loaded {len(samples)} samples")
    if args.component != "chain":
        # chain は label が dict なので by_label 集計が冗長になる → スキップ
        print(f"[phase_i] summary: {_summary(samples)}")
    if args.component == "score":
        metrics = _fine_tune_score(samples, args)
    elif args.component == "chain":
        metrics = _fine_tune_chain(samples, args)
    elif args.component == "cell_color":
        metrics = _fine_tune_cell_color(samples, args)
    else:
        metrics = _fine_tune_next(samples, args)
    print(f"[phase_i] fine_tune metrics: {metrics}")


if __name__ == "__main__":
    main()
