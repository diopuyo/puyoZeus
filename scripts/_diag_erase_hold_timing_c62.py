"""真因計装: 連鎖「消去演出(気絶顔ホールド)」の実測秒数と誤STABLE化の実証 (2026-07-24)。

完全 read-only 診断スクリプト。src/ は一切変更しない
(recognition_pipeline.py / board_state_machine.py は読むだけ。
 別コーダが recognition_pipeline.py を直後に編集しているため絶対に触らない)。

背景 (col5 診断 scripts/_diag_col5_top_purple_c62_1p.py で確定した真因の疑い):
    連鎖の消去演出(ぷよが「気絶顔」で静止するホールド)フレームを、状態機械が
    STABLE と誤判定して confirmed_board に取り込んでいる可能性がある。
    stable_frame_count=6 (デフォルト、0.1s@60fps 相当) が、消去ポップの
    静止ホールド (数百ms?) より短ければ、CHAIN state を抜けて STABLE に
    戻った時点でまだ画面上は消去未完了 → confirmed_board に残像が焼き付く。

計測方法 (推測でなく実測で真因の材料を揃える):
    1. RecognitionPipeline を通常動作 (stable_frame_count=6、デフォルト設定)
       で走らせ、frame毎の state / confirmed_board / cnn_board / chain_event
       を記録する。
    2. side.chain_event が新規発火した瞬間、その直前の confirmed_board
       (= CHAIN 遷移前の最終 STABLE 盤面 = baseline) に対して
       ChainSimulator().find_erasable_groups(baseline) を実行し、
       「このステップで本来消えるはずの group」を特定する。
    3. 以降のフレームで、group の各セルの side.cnn_board 値を追跡し、
       いつ group.color から別の値 (消去完了) に変化するかを計測する
       (= 見た目上のホールド秒数の代理指標。CNN が「まだその色に見えている」
       期間 = 気絶顔ホールドを含む見た目上の消去未完了期間)。
    4. state 系列と突き合わせ、CHAIN を離脱して STABLE (または
       GRAVITY_SETTLE 経由で STABLE) に復帰した時刻が、上記の「消去完了
       時刻」より前なら「消去未完了なのに STABLE 確定した」誤 STABLE 化と
       判定する。

出力先: data/verify/recognition_diag_erase_hold_timing_c62/
    - erase_hold_events.json / .csv : イベント毎の詳細 (hold 秒数・
      誤STABLE有無等)
    - diff_timeline.png              : 盤面全体グレースケール diff + state
      + chain trigger 縦線のタイムライン
    - group_unerased_timeline.png    : 各追跡グループの「未消去セル比率」
      時系列 (state 背景色重畳)
    - strip_event_<n>_t<time>.png    : 代表イベントの group ROI 実画素
      連続フレームストリップ (気絶顔→消滅の推移)
    - summary.txt / summary.json     : ホールド秒数分布・誤STABLE件数・
      debounce 閾値候補

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/_diag_erase_hold_timing_c62.py
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

from src.board import Board, COLOR_EMPTY  # noqa: E402
from src.chain import ChainSimulator, PuyoGroup  # noqa: E402
from src.image_reader import BoardRegion, DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 定数
# ============================
VIDEO_PATH: str = "data/frames/video_c62.mp4"
PROC_START_SEC: float = 850.0   # state machine warmup 用マージン
DIAG_START_SEC: float = 890.0   # 記録開始 (疑似イベント群を包含)
DIAG_END_SEC: float = 965.0     # 記録終了

TARGET_SIDE: str = "1P"  # 主対象 (c62 game9 で問題が確認された side)

# group 追跡の最大観測窓 (秒)。この間に色変化が観測できなければ追跡打ち切り。
GROUP_TRACK_MAX_SEC: float = 4.0

# ストリップ viz を出す代表イベント数上限
STRIP_EVENT_LIMIT: int = 4
STRIP_MARGIN_PX: int = 24
STRIP_FRAME_STEP: int = 2  # 何フレーム毎に1コマ出すか (間引き)

OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "recognition_diag_erase_hold_timing_c62"


# ============================
# データ構造
# ============================


@dataclass
class _FrameRec:
    """1 frame・1 side 分の記録。"""

    frame_idx: int
    t: float
    state: str
    confirmed_board: Board | None
    cnn_board: Board
    chain_trigger_sec: float | None
    chain_count_event: int | None
    diff_luma: float  # 前 frame との盤面全体グレースケール diff (mean abs)


@dataclass
class _GroupTrack:
    """1 連鎖発火イベントに対する group 追跡結果。"""

    event_idx: int
    trigger_sec: float
    color: int
    cells: list[tuple[int, int]]
    group_size: int
    # 各セルが group.color から変化した時刻 (未観測なら None)
    cell_change_t: dict[tuple[int, int], float | None] = field(default_factory=dict)
    first_change_t: float | None = None   # 最初の 1 セルが変化した時刻
    all_change_t: float | None = None     # 全セルが変化した時刻
    chain_exit_t: float | None = None     # side.state が CHAIN でなくなった時刻
    stable_resume_t: float | None = None  # side.state が STABLE に復帰した時刻
    track_end_t: float | None = None      # 追跡終了時刻 (打ち切り含む)
    # 未消去セル比率の時系列 (viz 用): (t, fraction_unerased)
    unerased_series: list[tuple[float, float]] = field(default_factory=list)
    # アーキ追加要求 (2026-07-24): 汚染/偽イベント分離用。
    # contamination_confirmed: STABLE 復帰の瞬間、confirmed_board が
    # まだ tr.color (消去未完了色) を group cell に保持しているか
    # (= confirmed_board 実汚染の直接証拠)。
    contamination_confirmed: bool | None = None
    # その汚染色が confirmed_board 上で何秒間 (何frame) 持続したか
    # (次の state 遷移で上書きされる、または観測窓終了まで残るかを計測)。
    contamination_persist_sec: float | None = None


# ============================
# パス1: pipeline 走査 + group 追跡
# ============================


def _region_gray_crop(frame: np.ndarray, region: BoardRegion) -> np.ndarray:
    """盤面全体 (可視領域) のグレースケール crop を返す。"""
    x1, y1 = region.x, region.y
    x2, y2 = region.x + region.width, region.y + region.height
    h_img, w_img = frame.shape[:2]
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    patch = frame[y1:y2, x1:x2]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)


def _new_group_tracks(
    event_idx_counter: list[int],
    trigger_sec: float,
    baseline: Board | None,
    sim: ChainSimulator,
    pseudo_events: list[dict],
) -> list[_GroupTrack]:
    """chain_event 新規発火時、baseline から erasable groups を求め追跡開始する。

    アーキ追加要求 (2026-07-24): baseline に 4連結の erasable group が
    1つも見つからない場合、その発火は「盤面上に消去対象が無いのに
    chain_event が立った」偽/疑似イベントとみなし pseudo_events に記録する
    (confirmed_board を汚染する経路とは構造的に別、
     VideoChainTracker 側のスコア/カウントベース誤検出の疑い)。
    """
    if baseline is None:
        pseudo_events.append({"trigger_sec": trigger_sec, "reason": "no_baseline"})
        return []
    groups: list[PuyoGroup] = sim.find_erasable_groups(baseline)
    if not groups:
        pseudo_events.append({"trigger_sec": trigger_sec, "reason": "no_erasable_group_in_baseline"})
        return []
    tracks: list[_GroupTrack] = []
    for g in groups:
        event_idx_counter[0] += 1
        tracks.append(_GroupTrack(
            event_idx=event_idx_counter[0],
            trigger_sec=trigger_sec,
            color=g.color,
            cells=sorted(g.cells),
            group_size=g.size,
            cell_change_t={cell: None for cell in g.cells},
        ))
    return tracks


def _update_group_tracks(
    tracks: list[_GroupTrack], t: float, state: str, cnn_board: Board,
) -> None:
    """追跡中 group について cnn_board を見て変化検出・未消去比率を記録する。"""
    for tr in tracks:
        if tr.track_end_t is not None:
            continue
        n_unerased = 0
        for cell in tr.cells:
            r, c = cell
            if tr.cell_change_t[cell] is None:
                cur = int(cnn_board.get(r, c))
                if cur != tr.color:
                    tr.cell_change_t[cell] = t
                else:
                    n_unerased += 1
        tr.unerased_series.append((t, n_unerased / tr.group_size))
        changed_times = [v for v in tr.cell_change_t.values() if v is not None]
        if tr.first_change_t is None and changed_times:
            tr.first_change_t = min(changed_times)
        if tr.all_change_t is None and len(changed_times) == len(tr.cells):
            tr.all_change_t = max(changed_times)
        # state 記録: CHAIN 離脱 / STABLE 復帰の最初のタイミングを記録
        if tr.chain_exit_t is None and state != "chain" and t > tr.trigger_sec:
            tr.chain_exit_t = t
        if tr.stable_resume_t is None and state == "stable" and t > tr.trigger_sec:
            tr.stable_resume_t = t
        if t - tr.trigger_sec >= GROUP_TRACK_MAX_SEC:
            tr.track_end_t = t


def _collect(video_path: Path) -> tuple[list[_FrameRec], list[_GroupTrack], list[dict], float]:
    """video を warmup 付きで処理し、診断区間の frame 記録 + group 追跡を行う。"""
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    start_frame = int(PROC_START_SEC * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    end_frame = int(DIAG_END_SEC * fps)

    pipe = RecognitionPipeline.load_default(
        load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
    )
    pipe.set_video_id("c62")
    sim = ChainSimulator()
    region = DEFAULT_P1_REGION if TARGET_SIDE == "1P" else DEFAULT_P2_REGION

    frames: list[_FrameRec] = []
    tracks: list[_GroupTrack] = []
    pseudo_events: list[dict] = []
    event_idx_counter = [0]
    prev_gray: np.ndarray | None = None
    prev_trigger_sec: float | None = None
    prev_confirmed: Board | None = None

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
        side = r.p1 if TARGET_SIDE == "1P" else r.p2

        gray = _region_gray_crop(frame, region)
        diff_luma = 0.0
        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff_luma = float(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).mean())
        prev_gray = gray

        ce = side.chain_event
        trigger_sec = float(ce.trigger_sec) if ce is not None else None
        chain_count_event = int(ce.chain_count) if ce is not None else None

        if t >= DIAG_START_SEC:
            frames.append(_FrameRec(
                frame_idx=fi, t=t, state=side.state.name.lower(),
                confirmed_board=side.confirmed_board.copy() if side.confirmed_board else None,
                cnn_board=side.cnn_board.copy(), chain_trigger_sec=trigger_sec,
                chain_count_event=chain_count_event, diff_luma=diff_luma,
            ))
            # 新規発火検出 (trigger_sec が新しい値になった瞬間)
            if trigger_sec is not None and trigger_sec != prev_trigger_sec:
                new_tracks = _new_group_tracks(
                    event_idx_counter, trigger_sec, prev_confirmed, sim, pseudo_events,
                )
                tracks.extend(new_tracks)
            _update_group_tracks(tracks, t, side.state.name.lower(), side.cnn_board)

        if trigger_sec is not None:
            prev_trigger_sec = trigger_sec
        if side.confirmed_board is not None:
            prev_confirmed = side.confirmed_board.copy()

        fi += 1
        n_read += 1
        if n_read % 1200 == 0:
            print(f"[pass1] t={t:.1f}s まで処理済み ({n_read} frames)")
    cap.release()
    print(
        f"[pass1] 完了: {len(frames)} frame 記録、group追跡 {len(tracks)} 件、"
        f"pseudo_event {len(pseudo_events)} 件、fps={fps:.2f}"
    )
    return frames, tracks, pseudo_events, fps


# ============================
# 汚染判定 (アーキ追加要求 2026-07-24)
# ============================


def _annotate_contamination(tracks: list[_GroupTrack], frames: list[_FrameRec]) -> None:
    """STABLE 復帰の瞬間、confirmed_board が group.color を保持しているか
    (= confirmed_board の実汚染) を frame 記録から突き合わせて判定する。

    汚染確認後、その色が confirmed_board 上で何秒 (何 frame) 持続したかも
    測る (Phase C-7/E-1 の「STABLE 継続中は confirmed が再更新されない」
    仕様により、次の state 遷移までゴーストが残ることを実証するため)。
    """
    for tr in tracks:
        if tr.stable_resume_t is None:
            continue
        # stable_resume_t 以降で最初に confirmed_board が非 None になる frame を探す
        resume_frames = [fr for fr in frames if fr.t >= tr.stable_resume_t and fr.confirmed_board is not None]
        if not resume_frames:
            continue
        first = resume_frames[0]
        stale_cells = [
            cell for cell in tr.cells
            if int(first.confirmed_board.get(cell[0], cell[1])) == tr.color
        ]
        tr.contamination_confirmed = len(stale_cells) > 0
        if not tr.contamination_confirmed:
            tr.contamination_persist_sec = 0.0
            continue
        # 持続秒数: stale_cells のどれかが tr.color のままである最後の frame を探す
        persist_end_t = first.t
        for fr in resume_frames:
            if fr.confirmed_board is None:
                break
            still_stale = any(
                int(fr.confirmed_board.get(cell[0], cell[1])) == tr.color
                for cell in stale_cells
            )
            if still_stale:
                persist_end_t = fr.t
            else:
                break
        tr.contamination_persist_sec = persist_end_t - tr.stable_resume_t


# ============================
# CSV / JSON 出力
# ============================


def _write_frame_csv(frames: list[_FrameRec], out_path: Path) -> None:
    lines = ["t_sec,frame_idx,state,chain_trigger_sec,chain_count_event,diff_luma,puyo_count_confirmed"]
    for fr in frames:
        pc = fr.confirmed_board.count_puyos() if fr.confirmed_board is not None else -1
        lines.append(
            f"{fr.t:.3f},{fr.frame_idx},{fr.state},"
            f"{fr.chain_trigger_sec if fr.chain_trigger_sec is not None else ''},"
            f"{fr.chain_count_event if fr.chain_count_event is not None else ''},"
            f"{fr.diff_luma:.3f},{pc}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _track_to_dict(tr: _GroupTrack) -> dict:
    hold_sec = (
        tr.first_change_t - tr.trigger_sec if tr.first_change_t is not None else None
    )
    all_hold_sec = (
        tr.all_change_t - tr.trigger_sec if tr.all_change_t is not None else None
    )
    chain_dur_sec = (
        tr.chain_exit_t - tr.trigger_sec if tr.chain_exit_t is not None else None
    )
    false_stable = (
        tr.stable_resume_t is not None
        and tr.first_change_t is not None
        and tr.stable_resume_t < tr.first_change_t
    )
    false_stable_all = (
        tr.stable_resume_t is not None
        and tr.all_change_t is not None
        and tr.stable_resume_t < tr.all_change_t
    )
    return {
        "event_idx": tr.event_idx,
        "trigger_sec": tr.trigger_sec,
        "color": tr.color,
        "group_size": tr.group_size,
        "cells": tr.cells,
        "first_change_t": tr.first_change_t,
        "all_change_t": tr.all_change_t,
        "hold_sec_first_cell": hold_sec,
        "hold_sec_all_cells": all_hold_sec,
        "chain_exit_t": tr.chain_exit_t,
        "chain_state_duration_sec": chain_dur_sec,
        "stable_resume_t": tr.stable_resume_t,
        "false_stable_vs_first_change": false_stable,
        "false_stable_vs_all_change": false_stable_all,
        "track_truncated": tr.first_change_t is None,
        # アーキ追加要求 (2026-07-24): confirmed_board 実汚染の直接証拠。
        "contamination_confirmed": tr.contamination_confirmed,
        "contamination_persist_sec": tr.contamination_persist_sec,
    }


def _write_events(tracks: list[_GroupTrack], out_dir: Path) -> list[dict]:
    records = [_track_to_dict(tr) for tr in tracks]
    (out_dir / "erase_hold_events.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    cols = [
        "event_idx", "trigger_sec", "color", "group_size", "first_change_t",
        "all_change_t", "hold_sec_first_cell", "hold_sec_all_cells",
        "chain_exit_t", "chain_state_duration_sec", "stable_resume_t",
        "false_stable_vs_first_change", "false_stable_vs_all_change", "track_truncated",
        "contamination_confirmed", "contamination_persist_sec",
    ]
    lines = [",".join(cols)]
    for rec in records:
        lines.append(",".join(str(rec[c]) for c in cols))
    (out_dir / "erase_hold_events.csv").write_text("\n".join(lines), encoding="utf-8")
    return records


# ============================
# viz
# ============================


def _write_diff_timeline(frames: list[_FrameRec], tracks: list[_GroupTrack], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([fr.t for fr in frames])
    diff = np.array([fr.diff_luma for fr in frames])
    state_colors = {
        "stable": "#2ca02c", "chain": "#d62728", "gravity_settle": "#ff7f0e",
        "tsumo_fall": "#1f77b4", "ojama_fall": "#9467bd", "effect": "#7f7f7f",
        "menu": "#000000",
    }

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(t, diff, color="black", linewidth=0.7, label="盤面全体 grayscale diff (mean abs)")
    # state 背景帯
    for i in range(len(frames) - 1):
        c = state_colors.get(frames[i].state, "#cccccc")
        ax.axvspan(frames[i].t, frames[i + 1].t, color=c, alpha=0.15, linewidth=0)
    for tr in tracks:
        ax.axvline(tr.trigger_sec, color="blue", linestyle="--", alpha=0.5, linewidth=0.8)
        if tr.stable_resume_t is not None:
            ax.axvline(tr.stable_resume_t, color="green", linestyle=":", alpha=0.6, linewidth=0.8)
        if tr.first_change_t is not None:
            ax.axvline(tr.first_change_t, color="red", linestyle=":", alpha=0.6, linewidth=0.8)
    handles = [
        plt.Line2D([0], [0], color="black", lw=1, label="diff"),
        plt.Line2D([0], [0], color="blue", ls="--", label="chain trigger"),
        plt.Line2D([0], [0], color="green", ls=":", label="STABLE復帰"),
        plt.Line2D([0], [0], color="red", ls=":", label="group色変化(消去検出)"),
    ]
    for name, c in state_colors.items():
        handles.append(plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.15, label=f"state={name}"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("diff (0-255)")
    ax.set_title(f"{TARGET_SIDE} 盤面全体 diff + state 系列 + chain trigger/消去検出タイミング (c62 game9)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _write_group_unerased_timeline(tracks: list[_GroupTrack], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [tr for tr in tracks if tr.unerased_series]
    if not valid:
        return
    fig, axes = plt.subplots(len(valid), 1, figsize=(14, 2.2 * len(valid)), sharex=False)
    if len(valid) == 1:
        axes = [axes]
    for ax, tr in zip(axes, valid):
        ts = np.array([p[0] - tr.trigger_sec for p in tr.unerased_series])
        fr_ = np.array([p[1] for p in tr.unerased_series])
        ax.step(ts, fr_, where="post", color="#9467bd")
        ax.axvline(0, color="blue", linestyle="--", alpha=0.6, linewidth=0.8, label="trigger")
        if tr.stable_resume_t is not None:
            ax.axvline(tr.stable_resume_t - tr.trigger_sec, color="green", linestyle=":", linewidth=1.0, label="STABLE復帰")
        if tr.first_change_t is not None:
            ax.axvline(tr.first_change_t - tr.trigger_sec, color="red", linestyle=":", linewidth=1.0, label="初回色変化")
        ax.set_ylabel(f"#{tr.event_idx}\ncolor={tr.color}")
        ax.set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("経過時間 (trigger からの秒数)")
    fig.suptitle(f"{TARGET_SIDE} group毎「未消去セル比率」時系列 (c62 game9)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _group_bbox(cells: list[tuple[int, int]], region: BoardRegion) -> tuple[int, int, int, int]:
    """group の cells を包含する画面座標 bbox (x1,y1,x2,y2、余白込み) を返す。"""
    rows = [c[0] for c in cells]
    cols = [c[1] for c in cells]
    x1a, y1a, _, _ = region.cell_sample_rect(min(rows), min(cols))
    _, _, x2a, y2a = region.cell_sample_rect(max(rows), max(cols))
    return (
        x1a - STRIP_MARGIN_PX, y1a - STRIP_MARGIN_PX,
        x2a + STRIP_MARGIN_PX, y2a + STRIP_MARGIN_PX,
    )


def _write_event_strips(tracks: list[_GroupTrack], fps: float, video_path: Path, out_dir: Path) -> None:
    """代表イベントの group ROI を連続フレームで並べたストリップ画像を出す。"""
    region = DEFAULT_P1_REGION if TARGET_SIDE == "1P" else DEFAULT_P2_REGION
    # 追跡完了 (first_change_t 判明) したイベントを長め順に上位 N 件選ぶ
    candidates = [tr for tr in tracks if tr.first_change_t is not None]
    candidates.sort(key=lambda tr: (tr.first_change_t - tr.trigger_sec), reverse=True)
    picked = candidates[:STRIP_EVENT_LIMIT]
    if not picked:
        return
    cap = cv2.VideoCapture(str(video_path))
    for tr in picked:
        x1, y1, x2, y2 = _group_bbox(tr.cells, region)
        start_t = max(0.0, tr.trigger_sec - 0.1)
        end_t = tr.first_change_t + 0.4
        start_fi = int(round(start_t * fps))
        end_fi = int(round(end_t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_fi))
        tiles = []
        fi = start_fi
        while fi <= end_fi:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame.shape[:2] != (1080, 1920):
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            if (fi - start_fi) % STRIP_FRAME_STEP == 0:
                h_img, w_img = frame.shape[:2]
                xx1, yy1 = max(0, x1), max(0, y1)
                xx2, yy2 = min(w_img, x2), min(h_img, y2)
                tile = frame[yy1:yy2, xx1:xx2].copy()
                t_rel = fi / fps - tr.trigger_sec
                cv2.putText(
                    tile, f"{t_rel:+.2f}s", (4, 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA,
                )
                tiles.append(tile)
            fi += 1
        if not tiles:
            continue
        h = max(tl.shape[0] for tl in tiles)
        w = max(tl.shape[1] for tl in tiles)
        padded = []
        for tl in tiles:
            out = np.zeros((h, w, 3), dtype=np.uint8)
            out[: tl.shape[0], : tl.shape[1]] = tl
            padded.append(out)
        sep = np.full((h, 3, 3), (255, 255, 255), dtype=np.uint8)
        strip_parts = []
        for p in padded:
            strip_parts.append(p)
            strip_parts.append(sep)
        combo = np.hstack(strip_parts[:-1])
        label = f"{tr.trigger_sec:.2f}".replace(".", "_")
        cv2.imwrite(str(out_dir / f"strip_event_{tr.event_idx}_t{label}.png"), combo)
    cap.release()


# ============================
# summary
# ============================


def _format_summary(records: list[dict], pseudo_events: list[dict]) -> tuple[str, dict]:
    hold_vals = [r["hold_sec_first_cell"] for r in records if r["hold_sec_first_cell"] is not None]
    hold_all_vals = [r["hold_sec_all_cells"] for r in records if r["hold_sec_all_cells"] is not None]
    chain_dur_vals = [r["chain_state_duration_sec"] for r in records if r["chain_state_duration_sec"] is not None]
    n_false_first = sum(1 for r in records if r["false_stable_vs_first_change"])
    n_false_all = sum(1 for r in records if r["false_stable_vs_all_change"])
    n_truncated = sum(1 for r in records if r["track_truncated"])
    # アーキ追加要求 (2026-07-24): 汚染 (a) と偽イベント (b) の分離集計。
    # (a) 実群 (baseline に4連結あり) が見つかった上で、STABLE復帰時に
    #     confirmed_board が実際に消去未完了色を保持した件数 = 直接汚染。
    # (b) baseline に4連結が見つからず group 追跡すらできなかった発火
    #     = confirmed_board 更新経路に乗らない偽/疑似イベント (別経路)。
    n_contaminated = sum(1 for r in records if r.get("contamination_confirmed") is True)
    n_contamination_checked = sum(1 for r in records if r.get("contamination_confirmed") is not None)
    persist_vals = [
        r["contamination_persist_sec"] for r in records
        if r.get("contamination_confirmed") is True and r.get("contamination_persist_sec") is not None
    ]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "median": None, "max": None, "min": None}
        arr = np.array(vals)
        return {
            "n": len(vals), "median": float(np.median(arr)),
            "max": float(np.max(arr)), "min": float(np.min(arr)),
        }

    hold_stats = _stats(hold_vals)
    hold_all_stats = _stats(hold_all_vals)
    chain_dur_stats = _stats(chain_dur_vals)
    persist_stats = _stats(persist_vals)

    # debounce 閾値候補: 実測ホールド最大値 + マージン (経験的に +30%)
    margin_ratio = 1.3
    candidate_sec = hold_all_stats["max"] * margin_ratio if hold_all_stats["max"] else None
    n_pseudo = len(pseudo_events)
    n_real_events = len(records) + n_pseudo  # group追跡できた + pseudo (発火総数)
    summary_dict = {
        "n_events_tracked": len(records),
        "n_events_truncated_no_change_observed": n_truncated,
        "hold_sec_first_cell_stats": hold_stats,
        "hold_sec_all_cells_stats": hold_all_stats,
        "chain_state_duration_sec_stats": chain_dur_stats,
        "n_false_stable_vs_first_change": n_false_first,
        "n_false_stable_vs_all_change": n_false_all,
        "rate_false_stable_vs_all_change": (
            n_false_all / len(records) if records else None
        ),
        "debounce_candidate_sec_margin30pct": candidate_sec,
        # アーキ追加要求: 汚染 (a) vs 偽イベント (b) の分離。
        "n_fire_events_total": n_real_events,
        "n_events_with_real_erasable_group": len(records),
        "n_pseudo_events_no_erasable_group": n_pseudo,
        "rate_pseudo_events": n_pseudo / n_real_events if n_real_events else None,
        "n_events_contamination_checked": n_contamination_checked,
        "n_events_confirmed_board_contaminated": n_contaminated,
        "rate_confirmed_board_contaminated": (
            n_contaminated / n_contamination_checked if n_contamination_checked else None
        ),
        "contamination_persist_sec_stats": persist_stats,
    }
    lines = [
        "==== c62 game9 消去演出(気絶顔ホールド) 実測 + 誤STABLE化 実証 サマリ ====",
        f"追跡できたイベント数: {len(records)} (打ち切り={n_truncated})",
        f"ホールド秒数(最初の1セルが変化するまで): n={hold_stats['n']}"
        f" median={hold_stats['median']} max={hold_stats['max']} min={hold_stats['min']}",
        f"ホールド秒数(全セルが変化するまで): n={hold_all_stats['n']}"
        f" median={hold_all_stats['median']} max={hold_all_stats['max']} min={hold_all_stats['min']}",
        f"CHAIN state 滞留秒数 (trigger→state離脱): n={chain_dur_stats['n']}"
        f" median={chain_dur_stats['median']} max={chain_dur_stats['max']} min={chain_dur_stats['min']}",
        f"誤STABLE化件数 (STABLE復帰 < 初回色変化): {n_false_first}/{len(records)}",
        f"誤STABLE化件数 (STABLE復帰 < 全セル色変化): {n_false_all}/{len(records)}"
        f" (率={summary_dict['rate_false_stable_vs_all_change']})",
        f"debounce閾値候補 (実測max全セルホールド × 1.3マージン): {candidate_sec} 秒",
        "--- (a) vs (b) 分離集計 (アーキ追加要求) ---",
        f"発火総数: {n_real_events} (実group追跡{len(records)} + pseudo{n_pseudo})",
        f"(b) 偽イベント率 (baseline に4連結なし、confirmed汚染経路に乗らない): "
        f"{n_pseudo}/{n_real_events} = {summary_dict['rate_pseudo_events']}",
        f"(a) confirmed_board 実汚染件数 (STABLE復帰時にまだ消去未完了色が残存): "
        f"{n_contaminated}/{n_contamination_checked} = {summary_dict['rate_confirmed_board_contaminated']}",
        f"(a) 汚染色の confirmed_board 上での持続秒数: n={persist_stats['n']}"
        f" median={persist_stats['median']} max={persist_stats['max']}"
        " (次の state 遷移まで残存し続ける可能性を含む、要個別確認)",
        "--- 個別イベント (trigger_sec, hold_all_sec, chain_dur_sec, false_stable_all, contaminated) ---",
    ]
    for r in records:
        lines.append(
            f"  #{r['event_idx']} t={r['trigger_sec']:.2f} color={r['color']} "
            f"hold_all={r['hold_sec_all_cells']} chain_dur={r['chain_state_duration_sec']} "
            f"false_stable_all={r['false_stable_vs_all_change']} "
            f"contaminated={r.get('contamination_confirmed')}"
        )
    if pseudo_events:
        lines.append("--- pseudo_events (baselineに4連結なし、偽/疑似発火) ---")
        for pe in pseudo_events:
            lines.append(f"  t={pe['trigger_sec']:.2f} reason={pe['reason']}")
    return "\n".join(lines), summary_dict


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = PROJ_ROOT / VIDEO_PATH
    frames, tracks, pseudo_events, fps = _collect(video_path)
    _annotate_contamination(tracks, frames)

    _write_frame_csv(frames, OUTPUT_DIR / "frame_trace_1p.csv")
    records = _write_events(tracks, OUTPUT_DIR)
    _write_diff_timeline(frames, tracks, OUTPUT_DIR / "diff_timeline.png")
    _write_group_unerased_timeline(tracks, OUTPUT_DIR / "group_unerased_timeline.png")
    _write_event_strips(tracks, fps, video_path, OUTPUT_DIR)

    text, summary_dict = _format_summary(records, pseudo_events)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    (OUTPUT_DIR / "pseudo_events.json").write_text(
        json.dumps(pseudo_events, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[DONE] 出力先: {OUTPUT_DIR}")
    print(text)


if __name__ == "__main__":
    main()
