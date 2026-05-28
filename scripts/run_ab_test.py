"""A/B 対照実験ランナー (Phase I 精度評価)。

B0/B1/B2/B3 の 4 仮説を独立評価し、各仮説の寄与度を数値化する。

仮説定義:
    B0: I1 のみ (= 現状 working tree / baseline)
    B1: M1 warmup guard のみ (STABLE 遷移直後 12 frame confirmed 凍結)
    B2: MAX_PUYO のみ (BG_FP_FORCE_MAX_PUYO 144 → 4)
    B3: B1 + B2 両適用

評価対象:
    8 動画 (v29/v40/v51/v57/v70/v89/v95/v97) の各 2 clip = 16 clip

使い方:
    python scripts/run_ab_test.py
    python scripts/run_ab_test.py --variants b0 b1  # 特定仮説のみ
    python scripts/run_ab_test.py --max-frames 3600  # 高速評価 (60fps × 60s)

出力:
    data/verify/stable_cell_acc/ab_b0_<datetime>.json
    data/verify/stable_cell_acc/ab_b1_<datetime>.json
    data/verify/stable_cell_acc/ab_b2_<datetime>.json
    data/verify/stable_cell_acc/ab_b3_<datetime>.json
    data/verify/stable_cell_acc/ab_summary_<datetime>.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加 (script 直接実行時)
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.recognition_pipeline import RecognitionPipeline

# ============================
# A/B 仮説定義
# ============================

# B2: MAX_PUYO の実験値 (= 元値 144 → 4)
_B2_MAX_PUYO_VALUE: int = 4
_B2_BASELINE_MAX_PUYO: int = 144  # 元値メモ

# 評価対象動画 ID (各 clip 2 本)
_DEFAULT_VIDEO_IDS: list[str] = [
    "v29_match2_156s",
    "v40_match7_125s",
    "v51_match2_97s",
    "v57_match2_100s",
    "v70_match2_113s",
    "v89_match3_95s",
    "v95_match15_99s",
    "v97_match11_96s",
]

# holdout 動画 (精度評価の本命)
_HOLDOUT_IDS: list[str] = [
    "v29_match2_156s",
    "v89_match3_95s",
]

# 仮説別 pipeline 構築パラメータ
_VARIANT_CONFIGS: dict[str, dict] = {
    "b0": {
        "description": "I1 のみ (現状 baseline)",
        "enable_warmup_guard": False,
        "bg_fp_force_max_puyo": _B2_BASELINE_MAX_PUYO,
    },
    "b1": {
        "description": "M1 warmup guard のみ (STABLE 遷移直後 12 frame 凍結)",
        "enable_warmup_guard": True,
        "bg_fp_force_max_puyo": _B2_BASELINE_MAX_PUYO,
    },
    "b2": {
        "description": "MAX_PUYO のみ (144 → 4)",
        "enable_warmup_guard": False,
        "bg_fp_force_max_puyo": _B2_MAX_PUYO_VALUE,
    },
    "b3": {
        "description": "B1 + B2 両適用",
        "enable_warmup_guard": True,
        "bg_fp_force_max_puyo": _B2_MAX_PUYO_VALUE,
    },
}

# ============================
# pipeline 構築
# ============================

def _make_pipeline(
    video_id: str,
    enable_warmup_guard: bool,
    bg_fp_force_max_puyo: int,
) -> RecognitionPipeline:
    """仮説別 pipeline を構築する。"""
    from scripts.measure_stable_cell_acc import _inject_hsv, _resolve_hsv_path
    pipe = RecognitionPipeline.load_default(
        force_in_match=True,
        enable_warmup_guard=enable_warmup_guard,
        bg_fp_force_max_puyo=bg_fp_force_max_puyo,
    )
    _inject_hsv(pipe, _resolve_hsv_path(video_id))
    return pipe


# ============================
# 評価実行
# ============================

def _run_variant(
    variant: str,
    video_ids: list[str],
    holdout_ids: list[str],
    max_frames: int,
    sample_interval: float,
    output_dir: Path,
) -> dict:
    """1 仮説の評価を実行し結果 dict を返す。"""
    from scripts.measure_stable_cell_acc import (
        _resolve_video_path,
        _process_video,
        _aggregate_stats,
        _compute_holdout_summary,
        _judge_pass_fail,
        DISAGREEMENT_OUTPUT_LIMIT,
    )

    cfg = _VARIANT_CONFIGS[variant]
    print(f"\n{'='*60}")
    print(f"[ab_test] 仮説 {variant.upper()}: {cfg['description']}")
    print(f"  enable_warmup_guard={cfg['enable_warmup_guard']}")
    print(f"  bg_fp_force_max_puyo={cfg['bg_fp_force_max_puyo']}")
    print(f"{'='*60}")

    # measure_stable_cell_acc の _process_video を使わず
    # pipeline 構築だけ差し替える形で処理する
    import cv2
    import numpy as np
    from src.board_state_machine import BoardState
    from src.board import COLOR_EMPTY, COLOR_UNKNOWN
    from scripts.measure_stable_cell_acc import (
        VideoStats,
        _open_capture,
        _eval_one_frame,
        _make_pipeline_hsv_only,
    )

    stats_list = []
    disagreements: list[dict] = []

    for vid in video_ids:
        vpath = _resolve_video_path(vid, None)
        if vpath is None:
            print(f"[ab_test] 動画未発見: {vid} → スキップ", file=sys.stderr)
            continue

        cap_info = _open_capture(vpath, max_frames, sample_interval)
        if cap_info is None:
            print(f"[ab_test] 動画を開けません: {vpath}", file=sys.stderr)
            continue

        cap, fps, n_target, interval_frames = cap_info

        # 仮説別 pipeline と HSV-only pipeline (正解ラベル用) を並列構築
        pipe_cnn = _make_pipeline(
            vid,
            enable_warmup_guard=cfg["enable_warmup_guard"],
            bg_fp_force_max_puyo=cfg["bg_fp_force_max_puyo"],
        )
        pipe_hsv = _make_pipeline_hsv_only(vid)

        is_holdout = vid in holdout_ids
        print(f"[ab_test] {variant} {vid}: fps={fps:.1f} target={n_target} holdout={is_holdout}")

        stats = VideoStats(video_id=vid, is_holdout=is_holdout)
        for fi in range(n_target):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            _eval_one_frame(
                vid, fi, fps, interval_frames, frame,
                pipe_cnn, pipe_hsv, stats, disagreements,
            )
            if fi % 500 == 0 and fi > 0:
                print(
                    f"  [progress] {fi}/{n_target} ({fi*100/max(n_target,1):.0f}%) "
                    f"agreed={stats.agreed_cells} total={stats.total_cells}"
                )
        cap.release()
        rate = stats.agreed_cells / stats.total_cells if stats.total_cells > 0 else 0.0
        print(
            f"[ab_test] {variant} {vid} 完了: stable={stats.stable_frame_count} "
            f"total={stats.total_cells} acc={rate:.4f}"
        )
        stats_list.append(stats)

    if not stats_list:
        print(f"[ab_test] {variant}: 処理動画ゼロ件", file=sys.stderr)
        return {"variant": variant, "error": "no videos processed"}

    agg = _aggregate_stats(stats_list)
    holdout_summary = _compute_holdout_summary(stats_list, holdout_ids)
    holdout_acc = holdout_summary.get("acc") if holdout_ids else None
    verdict, failures = _judge_pass_fail(
        overall_acc=agg["overall"]["acc"],
        per_color=agg["per_color"],
        holdout_acc=holdout_acc,
        stats_list=stats_list,
    )

    result = {
        "variant": variant,
        "description": cfg["description"],
        "config": cfg,
        **agg,
        "holdout_summary": holdout_summary,
        "disagreement_cells": disagreements[:DISAGREEMENT_OUTPUT_LIMIT],
        "disagreement_total": len(disagreements),
        "verdict": verdict,
        "failures": failures,
        "meta": {
            "videos": video_ids,
            "holdout": holdout_ids,
            "max_frames": max_frames,
            "sample_interval_sec": sample_interval,
        },
    }

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    out_path = output_dir / f"ab_{variant}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[ab_test] {variant} 結果保存: {out_path}")
    print(f"  overall acc={agg['overall']['acc']:.4f}  verdict={verdict}")
    if holdout_acc is not None:
        print(f"  holdout acc={holdout_acc:.4f}")

    return result


# ============================
# サマリ比較表
# ============================

def _print_comparison(results: list[dict], b0_result: dict | None) -> None:
    """4 仮説の比較表を標準出力に表示する。"""
    print("\n" + "=" * 70)
    print("A/B 対照実験 比較表")
    print("=" * 70)
    header = f"{'仮説':<6}  {'全体 acc':>10}  {'holdout acc':>12}  {'B0 比改善':>10}  {'verdict'}"
    print(header)
    print("-" * 70)

    b0_acc = b0_result["overall"]["acc"] if b0_result else None
    b0_hold = b0_result["holdout_summary"].get("acc") if b0_result else None

    for r in results:
        var = r.get("variant", "?")
        ov_acc = r.get("overall", {}).get("acc", None)
        ho_acc = r.get("holdout_summary", {}).get("acc", None)
        verdict = r.get("verdict", "?")

        acc_str = f"{ov_acc:.4f}" if ov_acc is not None else "  N/A "
        hold_str = f"{ho_acc:.4f}" if ho_acc is not None else "    N/A  "

        if b0_acc is not None and ov_acc is not None:
            delta = ov_acc - b0_acc
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "    N/A "

        print(f"{var:<6}  {acc_str:>10}  {hold_str:>12}  {delta_str:>10}  {verdict}")

    print("-" * 70)

    # 色別最低 acc
    print("\n[色別最低 acc 比較]")
    print(f"{'仮説':<6}", end="")
    colors = ["empty", "red", "blue", "green", "yellow", "purple", "ojama"]
    for c in colors:
        print(f"  {c[:5]:>5}", end="")
    print()
    for r in results:
        var = r.get("variant", "?")
        pc = r.get("per_color", {})
        print(f"{var:<6}", end="")
        for c in colors:
            v = pc.get(c, None)
            s = f"{v:.3f}" if v is not None else "  N/A"
            print(f"  {s:>5}", end="")
        print()

    print("\n[真因仮説判定]")
    if b0_acc is not None:
        b1_delta = None
        b2_delta = None
        b3_delta = None
        for r in results:
            acc = r.get("overall", {}).get("acc")
            if acc is None:
                continue
            if r.get("variant") == "b1":
                b1_delta = acc - b0_acc
            elif r.get("variant") == "b2":
                b2_delta = acc - b0_acc
            elif r.get("variant") == "b3":
                b3_delta = acc - b0_acc

        if b1_delta is not None:
            print(f"  B1 (warmup guard) 単独寄与: {b1_delta:+.4f}")
        if b2_delta is not None:
            print(f"  B2 (MAX_PUYO 絞込) 単独寄与: {b2_delta:+.4f}")
        if b1_delta is not None and b2_delta is not None and b3_delta is not None:
            interaction = b3_delta - b1_delta - b2_delta
            print(f"  B1+B2 合計寄与: {b3_delta:+.4f}  (相互作用: {interaction:+.4f})")
            # 真因判定
            dominant = "B1" if abs(b1_delta) > abs(b2_delta) else "B2"
            print(f"  → 主因仮説: {dominant} (寄与度が大きい方)")

    print("=" * 70)


# ============================
# CLI
# ============================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B 対照実験ランナー")
    p.add_argument(
        "--variants", nargs="+", default=list(_VARIANT_CONFIGS.keys()),
        choices=list(_VARIANT_CONFIGS.keys()),
        help="評価する仮説 (default: 全4仮説)",
    )
    p.add_argument(
        "--videos", type=str, default=None,
        help="動画 ID リスト (カンマ区切り)。省略時はデフォルト 8 動画。",
    )
    p.add_argument(
        "--max-frames", type=int, default=0,
        help="1 動画あたり最大処理フレーム数 (0=制限なし)。",
    )
    p.add_argument(
        "--sample-interval", type=float, default=1.0 / 30.0,
        help="認識処理間隔 (秒)。",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("data/verify/stable_cell_acc"),
        help="結果 JSON 出力ディレクトリ。",
    )
    return p.parse_args()


def main() -> int:
    """全仮説を評価し比較表を出力する。PASS なら 0、FAIL あれば 1。"""
    args = _parse_args()
    video_ids = (
        [v.strip() for v in args.videos.split(",") if v.strip()]
        if args.videos else _DEFAULT_VIDEO_IDS
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ab_test] 評価開始: variants={args.variants} videos={len(video_ids)}本")

    results: list[dict] = []
    b0_result: dict | None = None

    for variant in args.variants:
        r = _run_variant(
            variant=variant,
            video_ids=video_ids,
            holdout_ids=_HOLDOUT_IDS,
            max_frames=args.max_frames,
            sample_interval=args.sample_interval,
            output_dir=output_dir,
        )
        results.append(r)
        if variant == "b0":
            b0_result = r

    _print_comparison(results, b0_result)

    # サマリ JSON
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    summary_path = output_dir / f"ab_summary_{ts}.json"
    summary = {
        "variants": [
            {
                "variant": r.get("variant"),
                "description": r.get("description"),
                "overall_acc": r.get("overall", {}).get("acc"),
                "holdout_acc": r.get("holdout_summary", {}).get("acc"),
                "verdict": r.get("verdict"),
                "per_color": r.get("per_color", {}),
                "config": r.get("config", {}),
            }
            for r in results
        ],
        "meta": {
            "videos": video_ids,
            "holdout": _HOLDOUT_IDS,
            "max_frames": args.max_frames,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[ab_test] サマリ保存: {summary_path}")

    any_fail = any(r.get("verdict") == "FAIL" for r in results)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
