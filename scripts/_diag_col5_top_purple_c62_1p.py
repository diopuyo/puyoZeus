"""真因診断: c62 game9 1P col5(右端列) 上段の「空白→紫」誤認 (2026-07-23)。

完全 read-only 診断スクリプト。src/ および既存 scripts/ は一切変更しない
(特に src/recognition_pipeline.py は別コーダが案1 stale_hold を実装中のため
 絶対に触らない)。

背景 (依頼元の要約、#44 mismatch_cell_dump_902_67.json で裏付け済み):
    c62 game9 1P, t=902.67 付近で row1,col5 / row2,col5 が
    expected_color=0 (空) に対し actual_confirmed_color=5 (紫) と誤認。
    cnn_raw=5 / hsv_only=5 で specular_suspect=false
    (= 光沢ハイライト誤読ではない本物の false-positive)。

検証したい主仮説:
    この右上「空白→紫」誤認の点滅が、#44 で判明した「偽連鎖イベント乱発
    (event_score=0・chain_count=1 決め打ち)」および「1P だけ乱発・2P は
    ゼロの非対称」の共通原因ではないか。

出力先: data/verify/recognition_diag_col5_top_purple_c62_1p/
    - col5_top_trace_1p.csv / col5_top_trace_2p.csv : frame毎 confirmed/cnn/HSV
    - col5_top_timeline.png                          : 1P/2P 比較タイムライン
    - flip_event_correlation.json                    : 点滅時刻と偽イベント時刻の突合
    - crop_<side>_t<time>.png                          : col5 上段 ROI crop (実画素)
    - summary.txt / summary.json                       : 結論

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/_diag_col5_top_purple_c62_1p.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# 熱対策 (feedback_thermal_safety_mandatory 準拠)。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import BOARD_COLS, HIDDEN_ROWS  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 定数
# ============================
VIDEO_PATH: str = "data/frames/video_c62.mp4"
PROC_START_SEC: float = 850.0   # state machine warmup 用マージン
DIAG_START_SEC: float = 895.0   # 依頼区間
DIAG_END_SEC: float = 960.0

# 対象セル: 列5 (右端、0-indexed)、上段可視 row1/row2 (HIDDEN_ROWS=1 の次)。
TARGET_COL: int = 5
TARGET_ROWS: tuple[int, ...] = (1, 2)  # 可視最上段2行 (row0 は隠し段=画像なし)

COLOR_EMPTY_VAL: int = 0
COLOR_PURPLE_VAL: int = 5

# 依頼本文に列挙された疑似イベント時刻 (#44 summary.txt: event_score=0, cc=1 相当)。
PSEUDO_EVENT_TIMES_SEC: tuple[float, ...] = (
    901.68, 903.05, 917.92, 918.63, 918.73,
    929.78, 934.73, 936.10, 937.45, 949.30,
)
# 相関判定の許容窓 (秒): 点滅が発火前後この範囲内にあれば「相関あり」とみなす。
FLIP_CORRELATION_WINDOW_SEC: float = 1.0

# crop 出力用の余白 (px)。col5 セル自体 (幅64) に加えて隣接文脈を見せる。
CROP_MARGIN_PX: int = 40
# crop を出す代表時刻 (1P): 誤認 (紫) 中の瞬間・直後の解消瞬間・chain と無関係な平時。
SAMPLE_CROP_TIMES_1P: tuple[float, ...] = (896.0, 902.4, 902.67, 902.9, 903.3, 917.9, 918.6)
# 2P 側は同時刻で比較 crop (非対称確認用)。
SAMPLE_CROP_TIMES_2P: tuple[float, ...] = SAMPLE_CROP_TIMES_1P

OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "recognition_diag_col5_top_purple_c62_1p"


# ============================
# データ構造
# ============================


@dataclass
class _CellRec:
    """1 frame・1 セル分の観測値。"""

    frame_idx: int
    t: float
    state: str
    row: int
    col: int
    confirmed_color: int  # -1 = confirmed_board が None (非STABLE)
    cnn_color: int
    h: int
    s: int
    v: int


@dataclass
class _SideFrame:
    """1 frame・1 side 分のまとめ (相関解析・crop 選定用)。"""

    frame_idx: int
    t: float
    state: str
    cell_recs: list[_CellRec]
    chain_trigger_sec: float | None = None
    chain_count_event: int | None = None
    chain_total_score: int | None = None


def _cell_hsv_median(frame: np.ndarray, region, row: int, col: int) -> tuple[int, int, int]:
    """1 cell の HSV 中央値 (h, s, v) を frame から直接計算する (reader 内部状態に非依存)。"""
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    h_img, w_img = frame.shape[:2]
    x1, x2 = max(0, min(x1, w_img - 1)), max(x1 + 1, min(x2, w_img))
    y1, y2 = max(0, min(y1, h_img - 1)), max(y1 + 1, min(y2, h_img))
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return -1, -1, -1
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return (
        int(np.median(hsv[:, :, 0])),
        int(np.median(hsv[:, :, 1])),
        int(np.median(hsv[:, :, 2])),
    )


# ============================
# パス1: pipeline 走査
# ============================


def _collect_side_frames() -> tuple[list[_SideFrame], list[_SideFrame], float]:
    """video を warmup 付きで処理し、診断区間の 1P/2P col5 上段データを集める。"""
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(PROJ_ROOT / VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    start_frame = int(PROC_START_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    end_frame = int(DIAG_END_SEC * fps)

    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    pipe.set_video_id("c62")

    frames_1p: list[_SideFrame] = []
    frames_2p: list[_SideFrame] = []
    fi = start_frame
    n_read = 0
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        if t >= DIAG_START_SEC:
            frames_1p.append(_build_side_frame(fi, t, r.p1, frame, DEFAULT_P1_REGION))
            frames_2p.append(_build_side_frame(fi, t, r.p2, frame, DEFAULT_P2_REGION))
        fi += 1
        n_read += 1
        if n_read % 1200 == 0:
            print(f"[pass1] t={t:.1f}s まで処理済み ({n_read} frames)")
    cap.release()
    print(f"[pass1] 完了: 1P/2P各 {len(frames_1p)} frame 記録 (fps={fps:.2f})")
    return frames_1p, frames_2p, fps


def _build_side_frame(
    fi: int, t: float, side: object, frame: np.ndarray, region: object,
) -> _SideFrame:
    """SideResult 1件から col5 上段の _SideFrame を組み立てる。"""
    cell_recs: list[_CellRec] = []
    for row in TARGET_ROWS:
        confirmed_color = -1
        if side.confirmed_board is not None:
            confirmed_color = int(side.confirmed_board.get(row, TARGET_COL))
        cnn_color = int(side.cnn_board.get(row, TARGET_COL))
        h, s, v = _cell_hsv_median(frame, region, row, TARGET_COL)
        cell_recs.append(_CellRec(
            frame_idx=fi, t=t, state=side.state.name, row=row, col=TARGET_COL,
            confirmed_color=confirmed_color, cnn_color=cnn_color, h=h, s=s, v=v,
        ))
    ce = side.chain_event
    return _SideFrame(
        frame_idx=fi, t=t, state=side.state.name, cell_recs=cell_recs,
        chain_trigger_sec=float(ce.trigger_sec) if ce is not None else None,
        chain_count_event=int(ce.chain_count) if ce is not None else None,
        chain_total_score=int(ce.total_score) if ce is not None else None,
    )


# ============================
# 点滅 (flip) 検出
# ============================


@dataclass
class _Flip:
    """空↔紫 (または他色) の遷移 1 件。"""

    t: float
    frame_idx: int
    row: int
    from_color: int
    to_color: int
    source: str  # "confirmed" or "cnn"


def _detect_flips(frames: list[_SideFrame], source: str) -> list[_Flip]:
    """col5 の各 row で from_color!=to_color な遷移を検出する (confirmed_color=-1 は無視しない、
    非STABLE 中も継続的に「認識上どう見えているか」を追うため素通しする)。
    """
    flips: list[_Flip] = []
    last: dict[int, int] = {}
    for sf in frames:
        for cr in sf.cell_recs:
            val = cr.confirmed_color if source == "confirmed" else cr.cnn_color
            prev = last.get(cr.row)
            if prev is not None and prev != val:
                flips.append(_Flip(
                    t=cr.t, frame_idx=cr.frame_idx, row=cr.row,
                    from_color=prev, to_color=val, source=source,
                ))
            last[cr.row] = val
    return flips


def _purple_flip_rate(flips: list[_Flip]) -> int:
    """紫がらみ (from か to のどちらかが紫) の遷移件数。"""
    return sum(
        1 for f in flips
        if f.from_color == COLOR_PURPLE_VAL or f.to_color == COLOR_PURPLE_VAL
    )


# ============================
# 偽イベントとの相関
# ============================


def _nearest_flip_distance(flips: list[_Flip], t_event: float) -> tuple[float | None, float | None]:
    """イベント時刻に最も近い (紫がらみ) flip の時刻と距離 (符号付き: flip - event) を返す。"""
    purple_flips = [
        f for f in flips
        if f.from_color == COLOR_PURPLE_VAL or f.to_color == COLOR_PURPLE_VAL
    ]
    if not purple_flips:
        return None, None
    best = min(purple_flips, key=lambda f: abs(f.t - t_event))
    return best.t, best.t - t_event


def _correlate_with_pseudo_events(
    flips_1p_confirmed: list[_Flip], flips_1p_cnn: list[_Flip],
) -> dict:
    """依頼本文の疑似イベント時刻と col5 紫flip の時間相関を計算する。"""
    results = []
    for t_evt in PSEUDO_EVENT_TIMES_SEC:
        nearest_conf_t, delta_conf = _nearest_flip_distance(flips_1p_confirmed, t_evt)
        nearest_cnn_t, delta_cnn = _nearest_flip_distance(flips_1p_cnn, t_evt)
        correlated_conf = delta_conf is not None and abs(delta_conf) <= FLIP_CORRELATION_WINDOW_SEC
        correlated_cnn = delta_cnn is not None and abs(delta_cnn) <= FLIP_CORRELATION_WINDOW_SEC
        results.append({
            "t_pseudo_event": t_evt,
            "nearest_purple_flip_confirmed_t": nearest_conf_t,
            "delta_confirmed_sec": delta_conf,
            "correlated_confirmed": correlated_conf,
            "nearest_purple_flip_cnn_t": nearest_cnn_t,
            "delta_cnn_sec": delta_cnn,
            "correlated_cnn": correlated_cnn,
        })
    n_correlated_conf = sum(1 for r in results if r["correlated_confirmed"])
    n_correlated_cnn = sum(1 for r in results if r["correlated_cnn"])
    return {
        "window_sec": FLIP_CORRELATION_WINDOW_SEC,
        "n_pseudo_events": len(PSEUDO_EVENT_TIMES_SEC),
        "n_correlated_confirmed": n_correlated_conf,
        "n_correlated_cnn": n_correlated_cnn,
        "rate_correlated_confirmed": n_correlated_conf / len(PSEUDO_EVENT_TIMES_SEC),
        "rate_correlated_cnn": n_correlated_cnn / len(PSEUDO_EVENT_TIMES_SEC),
        "per_event": results,
    }


# ============================
# viz: タイムライン
# ============================


def _write_timeline_plot(
    frames_1p: list[_SideFrame], frames_2p: list[_SideFrame], out_path: Path,
) -> None:
    """1P/2P の col5 row1/row2 confirmed_color 系列 + 疑似イベント縦線を描画する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True)
    panels = [
        (frames_1p, 1, "1P row1(col5)"), (frames_1p, 2, "1P row2(col5)"),
        (frames_2p, 1, "2P row1(col5)"), (frames_2p, 2, "2P row2(col5)"),
    ]
    for ax, (frames, row, label) in zip(axes, panels):
        t = np.array([sf.t for sf in frames])
        vals = np.array([
            next(cr.confirmed_color for cr in sf.cell_recs if cr.row == row)
            for sf in frames
        ])
        is_purple = vals == COLOR_PURPLE_VAL
        is_empty = vals == COLOR_EMPTY_VAL
        is_none = vals == -1
        is_other = ~is_purple & ~is_empty & ~is_none
        ax.scatter(t[is_empty], vals[is_empty], c="#2ca02c", s=8, marker="s", label="empty(0)")
        ax.scatter(t[is_purple], vals[is_purple], c="#9467bd", s=8, marker="s", label="purple(5)")
        ax.scatter(t[is_other], vals[is_other], c="#1f77b4", s=8, marker="s", label="other")
        ax.scatter(t[is_none], np.full(is_none.sum(), -1), c="#d62728", s=8, marker="s", label="None(非STABLE)")
        for t_evt in PSEUDO_EVENT_TIMES_SEC:
            ax.axvline(t_evt, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_ylabel(label)
        ax.set_ylim(-2, 10.5)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].set_title(
        "col5(右端) 上段 confirmed_color 時系列 (点線=依頼本文の疑似イベント時刻)",
    )
    axes[-1].set_xlabel("time (sec)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ============================
# viz: ROI crop (実画素)
# ============================


def _seek_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def _crop_col5_top(frame: np.ndarray, region, fps: float) -> np.ndarray:
    """col5 の row0(隠し段直下境界)〜row2 を含む領域を余白付きで crop する。"""
    cell_w = int(region.cell_width)
    cell_h = int(region.cell_height)
    x1 = region.x + TARGET_COL * cell_w - CROP_MARGIN_PX
    x2 = region.x + region.width + CROP_MARGIN_PX  # 右端 (盤面外の隣接文脈も見る)
    y1 = region.y - CROP_MARGIN_PX  # 上端境界より上 (隠し段/次ぷよ等の文脈)
    y2 = region.y + 3 * cell_h + CROP_MARGIN_PX  # row0(仮想)〜row2 分
    h_img, w_img = frame.shape[:2]
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    crop = frame[y1:y2, x1:x2].copy()
    # グリッド線 (col5 境界・row1/row2 境界) を薄く重畳して座標を明示
    col5_x_local = region.x + TARGET_COL * cell_w - x1
    cv2.line(crop, (col5_x_local, 0), (col5_x_local, crop.shape[0]), (0, 255, 255), 1)
    for row_boundary in (0, 1, 2, 3):
        y_local = region.y + row_boundary * cell_h - y1
        if 0 <= y_local < crop.shape[0]:
            cv2.line(crop, (0, y_local), (crop.shape[1], y_local), (0, 255, 255), 1)
    return crop


def _write_sample_crops(fps: float) -> None:
    """代表時刻の col5 上段 crop を 1P/2P 両方出力する (実画素、認識非依存)。"""
    cap = cv2.VideoCapture(str(PROJ_ROOT / VIDEO_PATH))
    for side_label, times, region in (
        ("1p", SAMPLE_CROP_TIMES_1P, DEFAULT_P1_REGION),
        ("2p", SAMPLE_CROP_TIMES_2P, DEFAULT_P2_REGION),
    ):
        for t in times:
            fi = int(round(t * fps))
            frame = _seek_frame(cap, fi)
            if frame is None:
                continue
            crop = _crop_col5_top(frame, region, fps)
            label = f"{t:.2f}".replace(".", "_")
            cv2.putText(
                crop, f"{side_label} t={t:.2f}s", (6, 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imwrite(str(OUTPUT_DIR / f"crop_{side_label}_t{label}.png"), crop)
    cap.release()


def _write_side_by_side_strip(fps: float, times: tuple[float, ...]) -> None:
    """1P/2P を同時刻で横並びにした比較画像を出す (非対称確認用)。"""
    cap = cv2.VideoCapture(str(PROJ_ROOT / VIDEO_PATH))
    for t in times:
        fi = int(round(t * fps))
        frame = _seek_frame(cap, fi)
        if frame is None:
            continue
        crop_1p = _crop_col5_top(frame, DEFAULT_P1_REGION, fps)
        crop_2p = _crop_col5_top(frame, DEFAULT_P2_REGION, fps)
        h = max(crop_1p.shape[0], crop_2p.shape[0])
        w = max(crop_1p.shape[1], crop_2p.shape[1])

        def _pad(img: np.ndarray) -> np.ndarray:
            out = np.zeros((h, w, 3), dtype=np.uint8)
            out[: img.shape[0], : img.shape[1]] = img
            return out

        sep = np.full((h, 6, 3), (255, 255, 255), dtype=np.uint8)
        combo = np.hstack([_pad(crop_1p), sep, _pad(crop_2p)])
        cv2.putText(
            combo, f"LEFT=1P  RIGHT=2P  t={t:.2f}s", (6, h - 8),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
        )
        label = f"{t:.2f}".replace(".", "_")
        cv2.imwrite(str(OUTPUT_DIR / f"sidebyside_t{label}.png"), combo)
    cap.release()


# ============================
# CSV / summary 出力
# ============================


def _write_trace_csv(frames: list[_SideFrame], out_path: Path) -> None:
    lines = [
        "t_sec,frame_idx,state,row,col,confirmed_color,cnn_color,h,s,v,"
        "chain_trigger_sec,chain_count_event,chain_total_score",
    ]
    for sf in frames:
        for cr in sf.cell_recs:
            lines.append(
                f"{cr.t:.3f},{cr.frame_idx},{cr.state},{cr.row},{cr.col},"
                f"{cr.confirmed_color},{cr.cnn_color},{cr.h},{cr.s},{cr.v},"
                f"{sf.chain_trigger_sec if sf.chain_trigger_sec is not None else ''},"
                f"{sf.chain_count_event if sf.chain_count_event is not None else ''},"
                f"{sf.chain_total_score if sf.chain_total_score is not None else ''}"
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_1p, frames_2p, fps = _collect_side_frames()

    _write_trace_csv(frames_1p, OUTPUT_DIR / "col5_top_trace_1p.csv")
    _write_trace_csv(frames_2p, OUTPUT_DIR / "col5_top_trace_2p.csv")
    _write_timeline_plot(frames_1p, frames_2p, OUTPUT_DIR / "col5_top_timeline.png")

    flips_1p_confirmed = _detect_flips(frames_1p, "confirmed")
    flips_1p_cnn = _detect_flips(frames_1p, "cnn")
    flips_2p_confirmed = _detect_flips(frames_2p, "confirmed")
    flips_2p_cnn = _detect_flips(frames_2p, "cnn")

    corr = _correlate_with_pseudo_events(flips_1p_confirmed, flips_1p_cnn)
    (OUTPUT_DIR / "flip_event_correlation.json").write_text(
        json.dumps(corr, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )

    _write_sample_crops(fps)
    _write_side_by_side_strip(fps, SAMPLE_CROP_TIMES_1P)

    summary = {
        "n_frames_1p": len(frames_1p), "n_frames_2p": len(frames_2p),
        "n_flips_1p_confirmed_total": len(flips_1p_confirmed),
        "n_flips_1p_confirmed_purple_related": _purple_flip_rate(flips_1p_confirmed),
        "n_flips_1p_cnn_purple_related": _purple_flip_rate(flips_1p_cnn),
        "n_flips_2p_confirmed_total": len(flips_2p_confirmed),
        "n_flips_2p_confirmed_purple_related": _purple_flip_rate(flips_2p_confirmed),
        "n_flips_2p_cnn_purple_related": _purple_flip_rate(flips_2p_cnn),
        "pseudo_event_correlation_rate_confirmed": corr["rate_correlated_confirmed"],
        "pseudo_event_correlation_rate_cnn": corr["rate_correlated_cnn"],
        "flip_list_1p_confirmed_purple": [
            {"t": f.t, "row": f.row, "from": f.from_color, "to": f.to_color}
            for f in flips_1p_confirmed
            if f.from_color == COLOR_PURPLE_VAL or f.to_color == COLOR_PURPLE_VAL
        ],
        "flip_list_2p_confirmed_purple": [
            {"t": f.t, "row": f.row, "from": f.from_color, "to": f.to_color}
            for f in flips_2p_confirmed
            if f.from_color == COLOR_PURPLE_VAL or f.to_color == COLOR_PURPLE_VAL
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    text = _format_summary_text(summary)
    (OUTPUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(f"[DONE] 出力先: {OUTPUT_DIR}")
    print(text)


def _format_summary_text(summary: dict) -> str:
    lines = [
        "==== c62 game9 col5(右端)上段 空白→紫 誤認 真因診断 サマリ ====",
        f"1P confirmed 紫がらみ flip 件数: {summary['n_flips_1p_confirmed_purple_related']}"
        f" (全flip {summary['n_flips_1p_confirmed_total']})",
        f"1P cnn 紫がらみ flip 件数: {summary['n_flips_1p_cnn_purple_related']}",
        f"2P confirmed 紫がらみ flip 件数: {summary['n_flips_2p_confirmed_purple_related']}"
        f" (全flip {summary['n_flips_2p_confirmed_total']})",
        f"2P cnn 紫がらみ flip 件数: {summary['n_flips_2p_cnn_purple_related']}",
        f"疑似イベント時刻との相関率 (confirmed, window={FLIP_CORRELATION_WINDOW_SEC}s): "
        f"{summary['pseudo_event_correlation_rate_confirmed']:.2f}",
        f"疑似イベント時刻との相関率 (cnn, window={FLIP_CORRELATION_WINDOW_SEC}s): "
        f"{summary['pseudo_event_correlation_rate_cnn']:.2f}",
        "--- 1P 紫flip一覧 (confirmed) ---",
    ]
    for f in summary["flip_list_1p_confirmed_purple"]:
        lines.append(f"  t={f['t']:.2f} row={f['row']} {f['from']}->{f['to']}")
    lines.append("--- 2P 紫flip一覧 (confirmed) ---")
    for f in summary["flip_list_2p_confirmed_purple"]:
        lines.append(f"  t={f['t']:.2f} row={f['row']} {f['from']}->{f['to']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
