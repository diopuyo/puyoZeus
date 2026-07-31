"""真因診断: 試合開始のコールドスタート遅延 (失敗2、2026-07-24 user目視確定分)。

完全 read-only 診断スクリプト。src/ および既存 scripts/ は一切変更しない。

## 依頼背景
user が目視で確定した頻発失敗2:
    「試合が始まった直後、ぷよを数手認識せず、5手目くらいでようやく認識が
    追いつく。序盤の最初の数手の盤面が空/欠落のまま」

## 検証したい仮説
「序盤 AUC が最低 (0.36-0.37、project_indicator_win_eval_2026-07-05 追記)」
の真因が、実は序盤の指標そのものの限界ではなく「序盤の盤面が認識できて
いない」認識欠落だとすれば、認識改善で序盤評価も改善する可能性がある。

## コード読解で特定した機構 (実測前の設計理解)
SideResult.board_none_reason (recognition_pipeline.py:186-233, 3217-3266)
は confirmed_board=None の理由を "cold_start" / "menu_reset" /
"chain_hold_none" / "other" に分類する診断計装が既に実装済み (常時計算、
挙動には影響しない)。

重要な区別 (本スクリプトが可視化する肝):
    - "cold_start" = self._ever_had_confirmed_1p/2p が False の間、つまり
      「このパイプライン instance が一度も STABLE 確定していない」状態。
      これは pipe.reset() が呼ばれない限りクリアされない
      (recognition_pipeline.py:1600-1603)。
    - 本番の指標収集経路 (scripts/collect_indicators_v2.py) は 1 動画に
      つき 1 pipeline instance を構築し、試合境界で reset() を呼ばない
      (grep 確認済、2026-07-24)。つまり "cold_start" は動画内の
      **最初の試合だけ** で発生し、2 試合目以降の「毎試合ごとのコールド
      スタート」は "menu_reset" (is_match_active: False→True 遷移後、
      board_state_machine.py:480-488 の MENU 強制からの再確定待ち) として
      観測されるはずである。
    - user が「頻発」と述べているのは 2 試合目以降も含む毎試合現象のため、
      本スクリプトは "cold_start" と "menu_reset" の両方を「試合開始直後の
      未確定区間」として合算集計する。

## 診断対象
    - c62: game9 開始 (872.4s)、game10 開始 (949.5s) — 実測 score0 境界。
    - video_30/33/35/38: data/verify/match_boundaries_v5/video_*/matches.tsv
      の先頭 2 試合 (idx1, idx2) の start_sec。
      video_30 = project_indicator_win_eval_2026-07-05 で中盤終盤とも良好
      (対照)。video_33/35/38 = 同評価で中盤終盤も不振 (認識欠落が真因か
      どうかの当たりを付ける対象)。

各 game start について PRE_MARGIN_SEC 手前からフレッシュな
RecognitionPipeline instance で処理開始 (= 本番の「動画内 最初の試合」を
模擬する意図。2 試合目以降は本来 menu_reset 経路になるが、本スクリプトは
各試合を独立 instance で見ることで「試合開始直後に何が起きるか」という
現象そのもの (遅延秒数・手数) を測る。cold_start/menu_reset いずれのタグに
せよ、フレーム毎の confirmed_board 有無という観測量は変わらない)。

## 出力
data/verify/recognition_diag_coldstart_2026-07-24/
    - summary.json / summary.txt: game-start 毎の遅延サマリ + 全体分布
    - samples.csv: game-start サンプル毎の生データ
    - viz_<video>_t<start>.png: 代表例 (試合開始直後 数手分の実画面 vs confirmed)

Usage (WSL 経由):
    wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
      PYTHONPATH=. ./venv/bin/python scripts/_diag_place_coldstart_startup_2026-07-24.py"
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board_state_machine import BoardState  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.visualize_recognition import (  # noqa: E402
    P1_ROI_X, P1_ROI_Y, P2_ROI_X, P2_ROI_Y, ROI_H, ROI_W, draw_cell_overlay,
)

# ============================
# 定数
# ============================

# game start サンプル: (video_stem, [start_sec, ...], note)
GAME_STARTS: tuple[tuple[str, tuple[float, ...], str], ...] = (
    ("c62", (872.4, 949.5), "game9/game10 実測score0境界"),
    ("30", (153.0, 190.0), "idx1/idx2、良好AUC動画(対照)"),
    ("33", (186.0, 228.0), "idx1/idx2"),
    ("35", (183.0, 264.0), "idx1/idx2、序盤/中盤/終盤AUC不振動画"),
    ("38", (154.0, 230.0), "idx1/idx2、序盤/中盤/終盤AUC不振動画"),
)

PRE_MARGIN_SEC: float = 4.0   # 試合開始前からの warmup 助走
POST_MARGIN_SEC: float = 18.0  # 試合開始後の観測窓 (数手〜5手超をカバーする想定)

CROP_MARGIN_PX: int = 24
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "recognition_diag_coldstart_2026-07-24"

# 未確定区間とみなす board_none_reason (cold_start と menu_reset を合算)。
STARTUP_NONE_REASONS: frozenset[str] = frozenset({"cold_start", "menu_reset"})


def _video_path(video_stem: str) -> Path:
    return PROJ_ROOT / "data" / "frames" / f"video_{video_stem}.mp4"


# ============================
# データ構造
# ============================


@dataclass
class _FrameRec:
    frame_idx: int
    t: float
    state: str
    is_match_active: bool
    confirmed_present: bool
    board_none_reason: str | None
    cnn_grid: np.ndarray


@dataclass
class _StartupSample:
    video: str
    side: str
    t_game_start: float
    t_match_active_true: float | None  # is_match_active が True になった最初の時刻
    t_first_confirmed: float | None  # confirmed_board が非 None になった最初の時刻
    delay_sec_from_start: float | None
    delay_sec_from_match_active: float | None
    n_tsumo_fall_entries_before_confirmed: int  # 「見逃した手数」の proxy
    reason_hist: dict  # board_none_reason 別のフレーム数 ([t_game_start, +POST_MARGIN])
    n_frames_window: int
    n_frames_none_in_window: int
    coverage_rate: float  # 観測窓内で confirmed_board が非 None だった割合


# ============================
# パス1: 走査
# ============================


def _collect_records(
    video_stem: str, start_sec: float,
) -> tuple[list[_FrameRec], list[_FrameRec], float]:
    cv2.setNumThreads(1)
    video_path = _video_path(video_stem)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    window_start = max(0.0, start_sec - PRE_MARGIN_SEC)
    window_end = start_sec + POST_MARGIN_SEC
    start_frame = int(window_start * fps)
    end_frame = int(window_end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    # フレッシュな pipeline instance (= 「この試合が最初の試合」を模擬)。
    pipe = RecognitionPipeline.load_default()
    pipe.set_video_id(video_stem)

    recs_1p: list[_FrameRec] = []
    recs_2p: list[_FrameRec] = []
    fi = start_frame
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        for side_recs, side_res in ((recs_1p, r.p1), (recs_2p, r.p2)):
            side_recs.append(_FrameRec(
                frame_idx=fi, t=t, state=side_res.state.name,
                is_match_active=r.is_match_active,
                confirmed_present=side_res.confirmed_board is not None,
                board_none_reason=side_res.board_none_reason,
                cnn_grid=side_res.cnn_board._grid.copy(),
            ))
        fi += 1
    cap.release()
    return recs_1p, recs_2p, fps


# ============================
# 解析
# ============================


def _analyze_startup(
    video: str, side: str, t_game_start: float, records: list[_FrameRec],
) -> _StartupSample:
    window_recs = [r for r in records if r.t >= t_game_start]
    t_active: float | None = None
    for r in window_recs:
        if r.is_match_active:
            t_active = r.t
            break
    t_confirmed: float | None = None
    for r in window_recs:
        if r.confirmed_present:
            t_confirmed = r.t
            break
    delay_from_start = (
        (t_confirmed - t_game_start) if t_confirmed is not None else None
    )
    delay_from_active = (
        (t_confirmed - t_active)
        if (t_confirmed is not None and t_active is not None) else None
    )
    # 「見逃した手数」 proxy: t_game_start 〜 t_confirmed の間に TSUMO_FALL に
    # 入った回数 (state 遷移の立ち上がりで検出)。
    n_tsumo_entries = 0
    prev_state = None
    for r in window_recs:
        if t_confirmed is not None and r.t > t_confirmed:
            break
        if r.state == BoardState.TSUMO_FALL.name and prev_state != BoardState.TSUMO_FALL.name:
            n_tsumo_entries += 1
        prev_state = r.state

    reason_hist: dict[str, int] = {}
    n_none_in_window = 0
    obs_window = [
        r for r in window_recs if r.t <= t_game_start + POST_MARGIN_SEC
    ]
    for r in obs_window:
        if not r.confirmed_present:
            n_none_in_window += 1
            key = r.board_none_reason or "unknown_none"
            reason_hist[key] = reason_hist.get(key, 0) + 1
    coverage = (
        1.0 - n_none_in_window / len(obs_window) if obs_window else 0.0
    )

    return _StartupSample(
        video=video, side=side, t_game_start=t_game_start,
        t_match_active_true=t_active, t_first_confirmed=t_confirmed,
        delay_sec_from_start=delay_from_start,
        delay_sec_from_match_active=delay_from_active,
        n_tsumo_fall_entries_before_confirmed=n_tsumo_entries,
        reason_hist=reason_hist, n_frames_window=len(obs_window),
        n_frames_none_in_window=n_none_in_window, coverage_rate=coverage,
    )


# ============================
# viz
# ============================


def _roi_for_side(side: str) -> tuple[int, int]:
    return (P1_ROI_X, P1_ROI_Y) if side == "1P" else (P2_ROI_X, P2_ROI_Y)


def _crop_roi(frame: np.ndarray, roi_x: int, roi_y: int) -> np.ndarray:
    x1 = max(0, roi_x - CROP_MARGIN_PX)
    y1 = max(0, roi_y - CROP_MARGIN_PX)
    x2 = min(frame.shape[1], roi_x + ROI_W + CROP_MARGIN_PX)
    y2 = min(frame.shape[0], roi_y + ROI_H + CROP_MARGIN_PX)
    return frame[y1:y2, x1:x2].copy()


def _seek_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def _render_startup_strip(
    video: str, side: str, sample: _StartupSample, records: list[_FrameRec], fps: float,
) -> None:
    """試合開始直後の数手分 (実画面 + confirmed 有無ラベル) を並べた strip を出力する。"""
    cap = cv2.VideoCapture(str(_video_path(video)))
    roi_x, roi_y = _roi_for_side(side)
    # 試合開始から 0/3/6/9/12/15 秒後の 6 コマをサンプル (約5-6手をカバーする想定)。
    offsets = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)
    panels: list[np.ndarray] = []
    by_frame = {r.frame_idx: r for r in records}
    for off in offsets:
        t = sample.t_game_start + off
        fi = int(round(t * fps))
        frame = _seek_frame(cap, fi)
        if frame is None:
            continue
        crop = _crop_roi(frame, roi_x, roi_y)
        rec = by_frame.get(fi)
        label = "?"
        color = (0, 255, 255)
        if rec is not None:
            label = f"conf={'OK' if rec.confirmed_present else rec.board_none_reason}"
            color = (0, 200, 0) if rec.confirmed_present else (0, 0, 255)
        cv2.putText(
            crop, f"+{off:.0f}s {label}", (8, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55,
            color, 2, cv2.LINE_AA,
        )
        panels.append(crop)
    cap.release()
    if not panels:
        return
    sep = np.full((panels[0].shape[0], 6, 3), (255, 255, 255), dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, sep, p])
    label = f"{sample.t_game_start:.1f}".replace(".", "_")
    cv2.imwrite(str(OUTPUT_DIR / f"viz_{video}_{side}_t{label}.png"), out)


# ============================
# メイン
# ============================


def _print_progress(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_samples: list[_StartupSample] = []
    n_viz_done = 0

    for video_stem, starts, note in GAME_STARTS:
        for t_start in starts:
            _print_progress(
                f"[{video_stem}] game-start t={t_start:.1f}s 開始 ({note})",
            )
            t0 = time.time()
            recs_1p, recs_2p, fps = _collect_records(video_stem, t_start)
            elapsed = time.time() - t0
            _print_progress(
                f"[{video_stem}] t={t_start:.1f}s pass1完了 "
                f"({len(recs_1p)} frame, {elapsed:.1f}s)",
            )
            for side, recs in (("1P", recs_1p), ("2P", recs_2p)):
                sample = _analyze_startup(video_stem, side, t_start, recs)
                all_samples.append(sample)
                _print_progress(
                    f"  {side}: delay_from_start="
                    f"{sample.delay_sec_from_start} sec, "
                    f"tsumo_entries_missed={sample.n_tsumo_fall_entries_before_confirmed}, "
                    f"coverage={sample.coverage_rate:.2f}",
                )
                if n_viz_done < 12:
                    _render_startup_strip(video_stem, side, sample, recs, fps)
                    n_viz_done += 1

    summary = _build_summary(all_samples)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.txt").write_text(_format_summary_text(summary), encoding="utf-8")
    _write_samples_csv(all_samples, OUTPUT_DIR / "samples.csv")
    _print_progress(f"[DONE] 出力先: {OUTPUT_DIR}")
    print(_format_summary_text(summary))


def _write_samples_csv(samples: list[_StartupSample], out_path: Path) -> None:
    lines = [
        "video,side,t_game_start,t_match_active_true,t_first_confirmed,"
        "delay_sec_from_start,delay_sec_from_match_active,"
        "n_tsumo_fall_entries_before_confirmed,coverage_rate,n_frames_window,"
        "n_frames_none_in_window,reason_hist",
    ]
    for s in samples:
        lines.append(
            f"{s.video},{s.side},{s.t_game_start:.2f},{s.t_match_active_true},"
            f"{s.t_first_confirmed},{s.delay_sec_from_start},"
            f"{s.delay_sec_from_match_active},"
            f"{s.n_tsumo_fall_entries_before_confirmed},{s.coverage_rate:.4f},"
            f"{s.n_frames_window},{s.n_frames_none_in_window},"
            f"\"{json.dumps(s.reason_hist, ensure_ascii=False)}\"",
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _build_summary(samples: list[_StartupSample]) -> dict:
    delays = [s.delay_sec_from_start for s in samples if s.delay_sec_from_start is not None]
    tsumo_missed = [s.n_tsumo_fall_entries_before_confirmed for s in samples]
    coverage = [s.coverage_rate for s in samples]
    n_delay_gt_2s = sum(1 for d in delays if d > 2.0)
    n_missed_ge_2_hands = sum(1 for n in tsumo_missed if n >= 2)
    n_never_confirmed = sum(1 for s in samples if s.t_first_confirmed is None)
    per_video: dict[str, dict] = {}
    for s in samples:
        v = per_video.setdefault(s.video, {"delays": [], "tsumo_missed": [], "coverage": []})
        if s.delay_sec_from_start is not None:
            v["delays"].append(s.delay_sec_from_start)
        v["tsumo_missed"].append(s.n_tsumo_fall_entries_before_confirmed)
        v["coverage"].append(s.coverage_rate)
    per_video_summary = {
        v: {
            "n_samples": len(d["coverage"]),
            "delay_sec_mean": (float(np.mean(d["delays"])) if d["delays"] else None),
            "delay_sec_max": (float(np.max(d["delays"])) if d["delays"] else None),
            "tsumo_missed_mean": float(np.mean(d["tsumo_missed"])),
            "coverage_rate_mean": float(np.mean(d["coverage"])),
        }
        for v, d in per_video.items()
    }
    return {
        "n_samples_total": len(samples),
        "delay_sec_from_start_mean": (float(np.mean(delays)) if delays else None),
        "delay_sec_from_start_median": (float(np.median(delays)) if delays else None),
        "delay_sec_from_start_max": (float(np.max(delays)) if delays else None),
        "n_delay_gt_2sec": n_delay_gt_2s,
        "n_delay_gt_2sec_rate": (n_delay_gt_2s / len(delays) if delays else None),
        "tsumo_fall_entries_missed_mean": float(np.mean(tsumo_missed)),
        "n_missed_ge_2_hands": n_missed_ge_2_hands,
        "n_missed_ge_2_hands_rate": n_missed_ge_2_hands / len(samples) if samples else None,
        "n_never_confirmed_within_window": n_never_confirmed,
        "coverage_rate_mean": (float(np.mean(coverage)) if coverage else None),
        "per_video": per_video_summary,
    }


def _format_summary_text(summary: dict) -> str:
    lines = [
        "==== 試合開始コールドスタート遅延 (失敗2) 頻度診断サマリ (2026-07-24) ====",
        f"サンプル数 (video x game-start x side): {summary['n_samples_total']}",
        f"confirmed 初出までの遅延秒 平均/中央値/最大: "
        f"{summary['delay_sec_from_start_mean']} / "
        f"{summary['delay_sec_from_start_median']} / "
        f"{summary['delay_sec_from_start_max']}",
        f"遅延 > 2秒 の割合: {summary['n_delay_gt_2sec_rate']} "
        f"({summary['n_delay_gt_2sec']} 件)",
        f"見逃した手数 (TSUMO_FALL進入回数) 平均: "
        f"{summary['tsumo_fall_entries_missed_mean']}",
        f"見逃し手数 >=2手 の割合: {summary['n_missed_ge_2_hands_rate']} "
        f"({summary['n_missed_ge_2_hands']} 件)",
        f"観測窓 (試合開始+{POST_MARGIN_SEC:.0f}s) 内でも一度も confirmed に "
        f"ならなかったサンプル数: {summary['n_never_confirmed_within_window']}",
        f"観測窓内 confirmed カバレッジ率 平均: {summary['coverage_rate_mean']}",
        "--- 動画別 ---",
    ]
    for v, s in summary["per_video"].items():
        lines.append(
            f"  video_{v}: n={s['n_samples']} delay_mean={s['delay_sec_mean']} "
            f"delay_max={s['delay_sec_max']} tsumo_missed_mean="
            f"{s['tsumo_missed_mean']:.2f} coverage_mean={s['coverage_rate_mean']:.2f}",
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
