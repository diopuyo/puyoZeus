"""A-3 最終確認: おじゃま収支 (ojama_net_balance) の1P/2P瞬時対称性を
間引きなしの密記録で検証する (2026-08-13、docs/CROSS_CUTTING_AUDIT_2026-08-13.md P4)。

## 背景
横展開監査で「1P/2P非対称 (20動画中19動画で2P平均>1P平均)」の最有力仮説が
「記録タイミングの側別非同期」に絞り込まれた。
scripts/collect_boards_lean.py の実装を読むと:
  - _drive_ojama_accounting_lean() は毎処理フレーム 1 回 OjamaAccountingTracker.
    get_snapshot() を呼び、1P/2P 共通の単一 snapshot を作る。
  - _ojama_snapshot_to_own_perspective() は net_1p = snap.net_balance_capped,
    net_2p = -net_1p という「同じ数の符号反転」で計算するため、
    この関数が呼ばれた瞬間には net_1p + net_2p は代数的に厳密 0 のはず。
  - しかし npz へ実際に書き込まれる行は _process_side_lean() 内の
    _should_emit() が側ごと独立に判定する (「自分の盤面が前回書き込み時と
    変わったか」 dedup)。1P と 2P で STABLE/dedup 通過するタイミングが
    ずれるため、npz 上の 1P 行集合と 2P 行集合は異なる時刻の観測値になる。

## 本スクリプトの計装方針 (本体無変更)
collect_boards_lean.collect_lean() を呼び出しつつ、モジュールレベル関数
_drive_ojama_accounting_lean / _process_side_lean を monkeypatch して
以下 2 種のログを密に記録する (本体のロジック分岐は一切変更しない、
呼び出しをラップして記録するだけ):
  1. dense_log: 毎処理フレーム (間引き後、side dedup 前) の
     (t_sec, net_1p, net_2p, bstate_1p, bstate_2p, tsumo_count_1p/2p)。
     → 「同一瞬間の瞬時対称性」を直接検証する。
  2. emit_log: _process_side_lean が実際に acc へ 1 行 append したか
     (npz と同じ dedup 後の行集合を再現)。
     → 側ごとの emit タイミングのずれが、npz レベルの平均値非対称を
       どの程度再現するかを定量化する。

使い方:
    ./venv/bin/python -m scripts._diag_ojama_symmetry_dense_2026-08-13 \\
        --video data/frames/review_demo_2026-08-12.mp4 \\
        --start-sec 0 --max-sec 600 \\
        --out-json logs/diag_ojama_symmetry_dense_2026-08-13.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import scripts.collect_boards_lean as clb  # noqa: E402
from src.production_config import (  # noqa: E402
    COLLECT_ONLY_ADOPTED,
    RECOGNITION_ADOPTED,
)

# 瞬時対称性の許容誤差 (float32 丸め誤差を許容する上限、2026-08-13)。
# net_balance_capped は int の減算なので理論上は厳密 0 のはずだが、
# _LeanNpzAccumulator 保存時の float32 キャストと同じ精度で判定する。
SYMMETRY_EPS: float = 1e-6


def _build_collect_kwargs() -> dict:
    """production_config の採用フラグを collect_lean() の kwargs に変換する。

    RECOGNITION_ADOPTED + COLLECT_ONLY_ADOPTED は 148 動画本番収集
    (scripts/_regen148_orchestrator_2026-08-11.py) が実際に使っている構成
    そのもの (collect_flags() の中身)。--with-next / --enable-phantom-board-guard
    / --sample-interval 0 も本番収集と同一にする (再現条件を本番と一致させる、
    デバッガ規律)。normalize_fps_30 は collect_lean() 既定 True のまま
    (60fps 動画は stride-2 = 実効30fps、本番と同一)。
    """
    kwargs: dict = {}
    for f in RECOGNITION_ADOPTED + COLLECT_ONLY_ADOPTED:
        parts = f.flag.split()
        name = parts[0].lstrip("-").replace("-", "_")
        kwargs[name] = True if len(parts) == 1 else float(parts[1])
    kwargs["capture_next"] = True  # --with-next 相当
    kwargs["enable_phantom_board_guard"] = True
    kwargs["sample_interval_sec"] = 0.0  # --sample-interval 0 相当
    return kwargs


def _install_instrumentation() -> tuple[list, list]:
    """collect_boards_lean のモジュール関数を monkeypatch し、密ログ2種を仕込む。

    Returns:
        (dense_log, emit_log) 2つの list。run 中に追記されていく。
    """
    dense_log: list[dict] = []
    emit_log: list[dict] = []

    orig_drive = clb._drive_ojama_accounting_lean

    def wrapped_drive(
        tracker, state_p1, state_p2, prev_bstate_p1, prev_bstate_p2,
        p1, p2, tsumo_count_1p, tsumo_count_2p, t_sec,
    ):
        snap = orig_drive(
            tracker, state_p1, state_p2, prev_bstate_p1, prev_bstate_p2,
            p1, p2, tsumo_count_1p, tsumo_count_2p, t_sec,
        )
        net_1p, forecast_1p, net_2p, forecast_2p = (
            clb._ojama_snapshot_to_own_perspective(snap)
        )
        dense_log.append({
            "t_sec": float(t_sec),
            "net_1p": net_1p,
            "net_2p": net_2p,
            "forecast_1p": forecast_1p,
            "forecast_2p": forecast_2p,
            "bstate_1p": p1.state.value if p1.state is not None else None,
            "bstate_2p": p2.state.value if p2.state is not None else None,
            "tsumo_count_1p": tsumo_count_1p,
            "tsumo_count_2p": tsumo_count_2p,
        })
        return snap

    clb._drive_ojama_accounting_lean = wrapped_drive

    orig_process_side = clb._process_side_lean

    def wrapped_process_side(
        acc, state, side_label, board, bstate, score, video_id, t_sec,
        frame_idx, **kwargs,
    ):
        n_before = len(acc.grids)
        orig_process_side(
            acc, state, side_label, board, bstate, score, video_id, t_sec,
            frame_idx, **kwargs,
        )
        n_after = len(acc.grids)
        emit_log.append({
            "side": side_label,
            "t_sec": float(t_sec),
            "frame_idx": frame_idx,
            "bstate": bstate.value if bstate is not None else None,
            "emitted": n_after > n_before,
            "ojama_net_balance": kwargs.get("ojama_net_balance"),
        })

    clb._process_side_lean = wrapped_process_side

    return dense_log, emit_log


def _analyze_dense_symmetry(dense_log: list[dict]) -> dict:
    """瞬時対称性 (net_1p + net_2p == 0) を全フレームで検証する。"""
    n = len(dense_log)
    residuals = np.array(
        [row["net_1p"] + row["net_2p"] for row in dense_log], dtype=np.float64,
    )
    violations = [
        dense_log[i] | {"residual": float(residuals[i])}
        for i in range(n) if abs(residuals[i]) > SYMMETRY_EPS
    ]
    return {
        "n_frames_dense": n,
        "max_abs_residual": float(np.max(np.abs(residuals))) if n else None,
        "n_violations": len(violations),
        "violation_rate_pct": (100.0 * len(violations) / n) if n else None,
        "sample_violations": violations[:20],
    }


def _analyze_emit_asymmetry(emit_log: list[dict]) -> dict:
    """side別 emit (npz相当) の平均値非対称を再現・定量化する。"""
    rows_1p = [r for r in emit_log if r["side"] == "1P" and r["emitted"]]
    rows_2p = [r for r in emit_log if r["side"] == "2P" and r["emitted"]]
    vals_1p = np.array([r["ojama_net_balance"] for r in rows_1p], dtype=np.float64)
    vals_2p = np.array([r["ojama_net_balance"] for r in rows_2p], dtype=np.float64)
    mean_1p = float(np.mean(vals_1p)) if len(vals_1p) else None
    mean_2p = float(np.mean(vals_2p)) if len(vals_2p) else None
    residual_mean = (
        (mean_1p + mean_2p) if mean_1p is not None and mean_2p is not None else None
    )
    return {
        "n_emitted_1p": len(rows_1p),
        "n_emitted_2p": len(rows_2p),
        "emit_count_ratio_2p_over_1p": (
            len(rows_2p) / len(rows_1p) if len(rows_1p) else None
        ),
        "mean_net_1p_emitted": mean_1p,
        "mean_net_2p_emitted": mean_2p,
        "residual_mean_should_be_0_if_synced": residual_mean,
    }


def _analyze_cross_time_gap(dense_log: list[dict], emit_log: list[dict]) -> dict:
    """emit時刻ズレの実効サイズを測る: 各 side の emit 時刻を dense_log の
    直近時刻に対応づけ、もう片方の side が emit した直近時刻との時間差・
    その間の net 変化量を集計する。非同期仮説の「機序」を直接示す数値。
    """
    t_dense = np.array([r["t_sec"] for r in dense_log])
    emitted_1p_t = np.array(
        [r["t_sec"] for r in emit_log if r["side"] == "1P" and r["emitted"]],
    )
    emitted_2p_t = np.array(
        [r["t_sec"] for r in emit_log if r["side"] == "2P" and r["emitted"]],
    )
    if len(emitted_1p_t) == 0 or len(emitted_2p_t) == 0 or len(t_dense) == 0:
        return {"note": "insufficient data"}

    net_1p_dense = np.array([r["net_1p"] for r in dense_log])
    # 各 1P emit 時刻に最も近い 2P emit 時刻を探し、その時間差と、
    # dense_log から見た本来あるべき net_1p の変化量を計算する。
    gaps = []
    for t1 in emitted_1p_t:
        idx2 = np.searchsorted(emitted_2p_t, t1)
        cands = []
        if idx2 < len(emitted_2p_t):
            cands.append(emitted_2p_t[idx2])
        if idx2 > 0:
            cands.append(emitted_2p_t[idx2 - 1])
        if not cands:
            continue
        t2 = min(cands, key=lambda x: abs(x - t1))
        gap = abs(t1 - t2)
        # dense_log 上で t1 と t2 に対応する net_1p の値を線形補間で取得
        v_at_t1 = np.interp(t1, t_dense, net_1p_dense)
        v_at_t2 = np.interp(t2, t_dense, net_1p_dense)
        gaps.append({
            "t_gap_sec": float(gap),
            "net_1p_drift_over_gap": float(v_at_t1 - v_at_t2),
        })
    gap_arr = np.array([g["t_gap_sec"] for g in gaps])
    drift_arr = np.array([g["net_1p_drift_over_gap"] for g in gaps])
    return {
        "n_pairs": len(gaps),
        "mean_t_gap_sec": float(np.mean(gap_arr)) if len(gap_arr) else None,
        "median_t_gap_sec": float(np.median(gap_arr)) if len(gap_arr) else None,
        "max_t_gap_sec": float(np.max(gap_arr)) if len(gap_arr) else None,
        "mean_abs_net_drift_over_gap": (
            float(np.mean(np.abs(drift_arr))) if len(drift_arr) else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A-3 最終確認: おじゃま収支瞬時対称性の密記録診断",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--max-sec", type=float, default=600.0)
    parser.add_argument(
        "--out-json", type=Path,
        default=Path("logs/diag_ojama_symmetry_dense_2026-08-13.json"),
    )
    args = parser.parse_args()

    dense_log, emit_log = _install_instrumentation()

    kwargs = _build_collect_kwargs()
    print(f"[diag] collect_lean kwargs: {kwargs}", flush=True)
    n_rows = clb.collect_lean(
        args.video,
        Path("logs/_diag_ojama_symmetry_dense_2026-08-13_scratch.npz"),
        start_sec=args.start_sec,
        max_sec=args.max_sec,
        **kwargs,
    )
    print(f"[diag] collect_lean returned n_rows={n_rows}", flush=True)
    print(f"[diag] dense_log frames={len(dense_log)} emit_log rows={len(emit_log)}", flush=True)

    result = {
        "video": str(args.video),
        "start_sec": args.start_sec,
        "max_sec": args.max_sec,
        "collect_kwargs": kwargs,
        "dense_symmetry": _analyze_dense_symmetry(dense_log),
        "emit_asymmetry": _analyze_emit_asymmetry(emit_log),
        "cross_time_gap": _analyze_cross_time_gap(dense_log, emit_log),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[diag] wrote {args.out_json}")
    print(json.dumps({k: v for k, v in result.items() if k != "collect_kwargs"},
                      ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
