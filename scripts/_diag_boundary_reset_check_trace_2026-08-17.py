"""問1計装: _update_game_boundary / observe_visual_signal の内部判定を
フレーム単位で記録する (本体コード非変更・計装ラッパー)。

「is_active の False→True 立ち上がりが score-reset の近傍で確認できたか
(near_visual)」を直接ログし、なぜ全ての score-reset がフォールバック
(anomaly) に落ちたのかを機構レベルで特定する。

W22根治検証 (2026-08-17 追記): _reconcile_boundary_anomalies による事後
救済後の最終 near_visual (resolved_near_visual) も併記する。オンライン
判定 (near_visual) は score-reset が視覚信号より時系列で先着する限り
False のままになり得るが (単一フレーム順パスの原理的限界)、事後救済後は
c109 の実例で anomalies が 0 件になることを確認する。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.collect_boards_lean as clb  # noqa: E402

OUT_DIR = Path("data/verify/diag_boundary_2026-08-17")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO = Path("data/frames/video_c109.mp4")
START_SEC = 3590.0
MAX_SEC = 220.0

_rows: list[dict] = []
_orig_update_boundary = clb._update_game_boundary
_orig_observe = clb._SharedGameCounter.observe_visual_signal


def _patched_observe(self, is_active, t_sec):
    prev = self._prev_is_active
    _orig_observe(self, is_active, t_sec)
    if prev is False and is_active:
        _rows.append({
            "kind": "visual_rise_candidate",
            "t_sec": round(t_sec, 3),
            "side": "",
            "is_reset": "",
            "near_visual": "",
            "advanced": "",
            "game_idx_after": self.game_idx,
            "last_visual_rise_sec": self.last_visual_rise_sec,
        })


def _patched_update_boundary(state, score, shared=None, t_sec=0.0, side_label=None):
    prev_score = state.prev_score
    is_reset = (
        prev_score is not None and score is not None
        and prev_score - score >= clb.SCORE_RESET_THRESHOLD
    )
    game_idx_before = shared.game_idx if shared is not None else state.game_idx
    _orig_update_boundary(state, score, shared, t_sec, side_label)
    if is_reset:
        near_visual = None
        if shared is not None and shared.multisignal_mode:
            # W22根治: last_visual_rise_sec (advance_if_new の成否と無関係に
            # 記録される専用フィールド) と突合する。それでもここはオンライン
            # (フレーム順の単一パス) の判定であり、score-reset が視覚信号
            # より時系列で先着するケースでは依然 False になり得る。
            near_visual = (
                shared.last_visual_rise_sec is not None
                and abs(t_sec - shared.last_visual_rise_sec)
                <= clb.BOUNDARY_MULTISIGNAL_TOLERANCE_SEC
            )
        game_idx_after = shared.game_idx if shared is not None else state.game_idx
        _rows.append({
            "kind": "score_reset_check",
            "t_sec": round(t_sec, 3),
            "side": side_label,
            "is_reset": True,
            "near_visual": near_visual,
            "advanced": game_idx_after != game_idx_before,
            "game_idx_after": game_idx_after,
            "last_visual_rise_sec": (
                shared.last_visual_rise_sec if shared is not None else None
            ),
        })


def main() -> int:
    clb._SharedGameCounter.observe_visual_signal = _patched_observe
    clb._update_game_boundary = _patched_update_boundary
    out_npz = OUT_DIR / "c109_g43_reset_check.npz"
    # 前回実行の異常ファイルが残っていると「今回は0件のため書き出さない」
    # ケースと区別がつかず誤読を招くため、実行前に削除しておく。
    stale_anomaly_path = out_npz.with_name(out_npz.stem + "_boundary_anomalies.json")
    stale_anomaly_path.unlink(missing_ok=True)
    n = clb.collect_lean(
        VIDEO, out_npz,
        start_sec=START_SEC, max_sec=MAX_SEC,
        enable_boundary_multisignal=True,
    )
    print(f"snapshots={n}", flush=True)

    out_csv = OUT_DIR / "c109_g43_reset_check_trace.csv"
    fieldnames = [
        "kind", "t_sec", "side", "is_reset", "near_visual", "advanced",
        "game_idx_after", "last_visual_rise_sec",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_rows)
    print(f"rows={len(_rows)} -> {out_csv}", flush=True)
    for r in _rows:
        print(r, flush=True)

    # W22根治検証: 動画処理完了後 (collect_lean 内部で既に実行済み) の
    # anomalies 件数を、written JSON ファイルから直接読み直して報告する
    # (_reconcile_boundary_anomalies は collect_lean 内で cap.release() 直後
    # 自動的に呼ばれるため、ここでは最終結果を確認するだけでよい)。
    anomaly_path = out_npz.with_name(out_npz.stem + "_boundary_anomalies.json")
    if anomaly_path.exists():
        import json
        anomalies = json.loads(anomaly_path.read_text(encoding="utf-8"))
        print(f"RESOLVED_ANOMALIES={len(anomalies)} -> {anomaly_path}", flush=True)
        for a in anomalies:
            print(a, flush=True)
    else:
        print("RESOLVED_ANOMALIES=0 (ファイル未生成 = 異常イベントなし)", flush=True)

    n_score_reset_checks = sum(1 for r in _rows if r["kind"] == "score_reset_check")
    n_online_near_visual_true = sum(
        1 for r in _rows if r["kind"] == "score_reset_check" and r["near_visual"]
    )
    print(
        f"score_reset_checks={n_score_reset_checks} "
        f"online_near_visual_true={n_online_near_visual_true}",
        flush=True,
    )
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
