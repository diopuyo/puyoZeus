"""多数決前の1フレームあたりセル分類精度 + 誤りの相関/独立判定 (2026-07-30)。

完全 read-only 診断スクリプト。src/ は一切変更しない (モンキーパッチも不要)。

## 測定対象 = 「投票前の生の分類結果」
SideResult.cnn_board (recognition_pipeline.py:225 「ImageReader 直の出力」)。
根拠:
  - 生成元は ImageReader.read_both_boards → read_board (単一フレーム処理のみ。
    UI mask / tier1 bg_fp / CNN+HSV 融合 / プロファイル / 浮遊除去 / 隠し段推論
    は全て同一フレーム内の防御層で、時系列の票は一切含まない)
  - temporal_smoothing=1 では cnn_1p = cnn_1p_raw (恒等、recognition_pipeline.py:2789)
  - 全ての票 (F ガード empty_to_color_min_votes / pending 多数決 stable_n /
    復旧ゲート / 着地票 / warmup) は cnn_board を消費して confirmed_board を
    作る側にあり、cnn_board 自体には作用しない

## 測定1: 1フレームあたりのセル不一致率
confirmed_board が STABLE で変化していない期間 (= run) において、
各フレームの cnn_board を run の confirmed_board と突き合わせる。
可視 12 段のみ比較 (隠し段 row0 は推論値なので除外)。
UNKNOWN セル (cnn 側=テロップ被覆等 / confirmed 側) は比較から除外して別カウント。

## 測定2: 誤りの相関/独立
同一セル・同一誤値の連続フレーム持続を「エピソード」として抽出し、
持続長分布を現在の票数 (3/8/18) と比較する。

## 汚染への防御 (層別で分離)
  - run 末尾: 次ツモ落下/連鎖開始の検知遅延中は cnn に実変化が写る
    → 「run 端からの距離」バケットで分離 + 中核区間 (端 trim) の値も併記
  - confirmed 自体の誤り (95.4% 測定): 相関型の一部は confirmed 側が誤り
    (未反映) の可能性 → 切り抜き画像で裏取りする

Usage (WSL 経由):
    PYTHONPATH=. nice -n 19 ./venv/bin/python \
        scripts/_diag_perframe_cell_accuracy_2026-07-30.py --targets c34:463:48:465.6
    PYTHONPATH=. ./venv/bin/python \
        scripts/_diag_perframe_cell_accuracy_2026-07-30.py --aggregate
    (--smoke で先頭ターゲット 8 秒のみの動作確認)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# CPU 競合対策: 330 ジョブ収集が稼働中のため 1 スレッドに固定する
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    HIDDEN_ROWS,
    VISIBLE_ROWS,
)
from src.image_reader import (  # noqa: E402
    DEFAULT_P1_REGION,
    DEFAULT_P2_REGION,
    BoardRegion,
)
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# ============================
# 定数
# ============================
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUT_DIR_DEFAULT: Path = PROJ_ROOT / "data" / "verify" / "perframe_accuracy_2026-07-30"

# 対象: stem:走査開始秒:走査秒数:試合開始秒 (試合開始前 2-3 秒は bg_fp 採取用)
# c34=30fps (game1 465.6-511.8s 既知)、c10/c19=60fps (boards_lean t_sec 密集区間)
DEFAULT_TARGETS: tuple[str, ...] = (
    "c34:463:48:465.6",
    "c10:166:50:168.6",
    "c19:147:50:149.6",
)

# 本番同値設定 (依頼指定)
STABLE_FRAME_COUNT: int = 3
TEMPORAL_SMOOTHING: int = 1
FORCE_IN_MATCH: bool = True

# run 採用の最短フレーム数 (これ未満の STABLE run は分析対象外として別カウント)
MIN_RUN_FRAMES: int = 10
# 中核区間 trim: run 先頭 (着地反映直後の残光) / 末尾 (次ツモ・連鎖の検知遅延)
HEAD_TRIM_FRAMES: int = 3
TAIL_TRIM_FRAMES: int = 6
# 相関型と分類する最短持続長 (= 最小の票数 3 に対応。感度は分布全体で併記)
CORRELATED_MIN_LEN: int = 3
# 現在の票数 (比較対象): F ガード3票 / フリッカ窓8 / STABLE CNN 履歴18
VOTE_WINDOWS: tuple[int, ...] = (3, 8, 18)
# run 端からの距離バケット (フレーム数、末尾汚染の分離用)
EDGE_BUCKET_BOUNDS: tuple[int, ...] = (2, 5, 10)  # 0-2 / 3-5 / 6-10 / 11+

# 切り抜き出力
CROPS_PER_TYPE: int = 6
CROP_MARGIN_CELLS: int = 2
CROP_SCALE: int = 2
CROP_RECT_COLOR: tuple[int, int, int] = (0, 0, 255)

PROGRESS_EVERY_FRAMES: int = 600
SMOKE_DUR_SEC: float = 8.0
VALID_COLORS: frozenset[int] = frozenset({1, 2, 3, 4, 5})


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================
# データ構造
# ============================


@dataclass
class _FrameRec:
    """1 (side, frame) 分の記録。"""

    frame_idx: int
    t: float
    state: str
    cnn: np.ndarray  # (13,6) int8: 投票前の生分類 (SideResult.cnn_board)
    confirmed: "np.ndarray | None"  # (13,6) int8 or None


@dataclass
class _SideAcc:
    """1 side 分の集計器。"""

    compared: int = 0                # 比較したセル×フレーム数
    unknown_cnn: int = 0             # cnn=UNKNOWN で除外
    unknown_conf: int = 0            # confirmed=UNKNOWN で除外
    by_type: dict = field(default_factory=dict)       # 型別 不一致フレーム数
    core_compared: int = 0           # 中核区間 (端 trim 後) の比較数
    core_by_type: dict = field(default_factory=dict)  # 中核区間の型別不一致
    cell_mismatch: np.ndarray = field(
        default_factory=lambda: np.zeros((VISIBLE_ROWS, BOARD_COLS), dtype=np.int64))
    cell_compared: np.ndarray = field(
        default_factory=lambda: np.zeros((VISIBLE_ROWS, BOARD_COLS), dtype=np.int64))
    edge_mismatch: dict = field(default_factory=dict)  # 末尾距離バケット別不一致
    edge_compared: dict = field(default_factory=dict)
    n_runs: int = 0
    run_frames: int = 0              # 採用 run の総フレーム数
    short_run_frames: int = 0        # MIN_RUN 未満で捨てた STABLE フレーム数
    state_counts: dict = field(default_factory=dict)  # 全フレームの state 分布


# ============================
# 走査 (pass 1)
# ============================


def _scan_video(
    stem: str, start_sec: float, dur_sec: float,
) -> tuple[list[_FrameRec], list[_FrameRec], float]:
    """動画を本番同値設定の pipeline で走査し 1P/2P の記録を返す。"""
    cv2.setNumThreads(1)
    path = VIDEO_DIR / f"video_{stem}.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fi = int(start_sec * fps)
    end_frame = int((start_sec + dur_sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=STABLE_FRAME_COUNT,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=TEMPORAL_SMOOTHING,
        force_in_match=FORCE_IN_MATCH,
    )
    pipe.set_video_id(stem)
    recs_1p: list[_FrameRec] = []
    recs_2p: list[_FrameRec] = []
    n = 0
    t0 = time.time()
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)
        for recs, sr in ((recs_1p, r.p1), (recs_2p, r.p2)):
            recs.append(_FrameRec(
                frame_idx=fi, t=t, state=sr.state.name,
                cnn=sr.cnn_board._grid.astype(np.int8),
                confirmed=(sr.confirmed_board._grid.astype(np.int8)
                           if sr.confirmed_board is not None else None),
            ))
        fi += 1
        n += 1
        if n % PROGRESS_EVERY_FRAMES == 0:
            rate = n / max(time.time() - t0, 1e-6)
            eta = time.strftime(
                "%H:%M", time.localtime(time.time() + (end_frame - fi) / max(rate, 1e-6)))
            _log(f"[{stem}] {n}f 処理済 ({rate:.1f}f/s) 完了見込み {eta} 頃")
    cap.release()
    return recs_1p, recs_2p, fps


# ============================
# STABLE run 分割
# ============================


def _segment_stable_runs(
    recs: list[_FrameRec], game_start_sec: float,
) -> tuple[list[tuple[int, int]], int]:
    """state==STABLE かつ confirmed 不変の連続区間 (a,b) 一覧を返す。

    game_start_sec 前のフレームは対象外 (force_in_match=True のため
    試合前のメニュー画面でも STABLE run が形成されうる)。
    戻り値: (runs, short_frames) short_frames=MIN_RUN 未満で捨てたフレーム数。
    """
    runs: list[tuple[int, int]] = []
    short_frames = 0
    a: int | None = None
    for i, rec in enumerate(recs):
        eligible = (
            rec.t >= game_start_sec
            and rec.state == "STABLE"
            and rec.confirmed is not None
        )
        cont = (
            eligible and a is not None
            and np.array_equal(rec.confirmed, recs[a].confirmed)
        )
        if cont:
            continue
        if a is not None:
            if i - a >= MIN_RUN_FRAMES:
                runs.append((a, i - 1))
            else:
                short_frames += i - a
        a = i if eligible else None
    if a is not None:
        if len(recs) - a >= MIN_RUN_FRAMES:
            runs.append((a, len(recs) - 1))
        else:
            short_frames += len(recs) - a
    return runs, short_frames


# ============================
# 不一致の型分類
# ============================


def _mismatch_type(conf_v: int, cnn_v: int) -> str:
    """confirmed の値と cnn の値から不一致の型ラベルを返す。"""
    if conf_v == COLOR_OJAMA or cnn_v == COLOR_OJAMA:
        if conf_v == COLOR_OJAMA and cnn_v == COLOR_EMPTY:
            return "ojama_to_empty"
        if conf_v == COLOR_OJAMA and cnn_v in VALID_COLORS:
            return "ojama_to_color"
        if conf_v in VALID_COLORS and cnn_v == COLOR_OJAMA:
            return "color_to_ojama"
        return "empty_to_ojama"
    if conf_v in VALID_COLORS and cnn_v == COLOR_EMPTY:
        return "color_to_empty"
    if conf_v in VALID_COLORS and cnn_v in VALID_COLORS:
        return "color_to_color"
    if conf_v == COLOR_EMPTY and cnn_v in VALID_COLORS:
        return "empty_to_color"
    return "other"


def _edge_bucket(dist_to_end: int) -> str:
    """run 末尾からの距離 (フレーム) をバケットラベルに変換する。"""
    for b in EDGE_BUCKET_BOUNDS:
        if dist_to_end <= b:
            return f"<= {b}"
    return f">{EDGE_BUCKET_BOUNDS[-1]}"


# ============================
# run 分析 (pass 1 の記録から)
# ============================


def _cell_value_episodes(
    vals: np.ndarray, v0: int,
) -> list[tuple[int, int, int]]:
    """1 セルの cnn 値時系列から (開始idx, 長さ, cnn値) の不一致エピソードを返す。

    「同一誤値の連続」を 1 エピソードとする (値が変われば別エピソード)。
    UNKNOWN は不一致に数えず、エピソードを切る。
    """
    eps: list[tuple[int, int, int]] = []
    cur_val: int | None = None
    cur_start = 0
    for i in range(len(vals)):
        v = int(vals[i])
        mism = (v != v0) and (v != COLOR_UNKNOWN)
        if mism and v == cur_val:
            continue
        if cur_val is not None:
            eps.append((cur_start, i - cur_start, cur_val))
            cur_val = None
        if mism:
            cur_val = v
            cur_start = i
    if cur_val is not None:
        eps.append((cur_start, len(vals) - cur_start, cur_val))
    return eps


def _analyze_run_frames(
    acc: _SideAcc, cnn_stack: np.ndarray, conf: np.ndarray, run_len: int,
) -> None:
    """run のフレームレベル集計 (比較数・型別・中核区間・末尾距離・セル位置)。"""
    vis = cnn_stack[:, HIDDEN_ROWS:, :]          # (L,12,6)
    conf_vis = conf[HIDDEN_ROWS:, :]             # (12,6)
    valid = conf_vis != COLOR_UNKNOWN            # confirmed=UNKNOWN セルは除外
    unk = (vis == COLOR_UNKNOWN) & valid[None]
    mism = (vis != conf_vis[None]) & ~unk & valid[None]
    acc.unknown_conf += int((~valid).sum()) * run_len
    acc.unknown_cnn += int(unk.sum())
    acc.compared += int(valid.sum()) * run_len - int(unk.sum())
    acc.cell_mismatch += mism.sum(axis=0)
    acc.cell_compared += valid.astype(np.int64) * run_len - unk.sum(axis=0)
    # 中核区間 (端 trim): run 先頭/末尾の汚染を除いた保守的な値
    lo, hi = HEAD_TRIM_FRAMES, run_len - TAIL_TRIM_FRAMES
    core_slice = slice(lo, hi) if hi > lo else slice(0, 0)
    acc.core_compared += (
        int(valid.sum()) * max(hi - lo, 0) - int(unk[core_slice].sum()))
    # 型別集計 (全体 + 中核)
    for f, r, c in np.argwhere(mism):
        mt = _mismatch_type(int(conf_vis[r, c]), int(vis[f, r, c]))
        acc.by_type[mt] = acc.by_type.get(mt, 0) + 1
        if lo <= f < hi:
            acc.core_by_type[mt] = acc.core_by_type.get(mt, 0) + 1
        b = _edge_bucket(run_len - 1 - int(f))
        acc.edge_mismatch[b] = acc.edge_mismatch.get(b, 0) + 1
    # 末尾距離バケットの分母 (フレームごとの比較セル数)
    per_frame_cmp = int(valid.sum()) - unk.sum(axis=(1, 2))
    for f in range(run_len):
        b = _edge_bucket(run_len - 1 - f)
        acc.edge_compared[b] = acc.edge_compared.get(b, 0) + int(per_frame_cmp[f])


def _analyze_run_episodes(
    recs: list[_FrameRec], a: int, b: int, video: str, side: str, run_id: int,
    cnn_stack: np.ndarray, conf: np.ndarray,
) -> list[dict]:
    """run 内の不一致エピソード (同一セル・同一誤値の連続) を抽出する。"""
    run_len = b - a + 1
    vis = cnn_stack[:, HIDDEN_ROWS:, :]
    conf_vis = conf[HIDDEN_ROWS:, :]
    valid = conf_vis != COLOR_UNKNOWN
    unk = (vis == COLOR_UNKNOWN) & valid[None]
    mism = (vis != conf_vis[None]) & ~unk & valid[None]
    episodes: list[dict] = []
    for r, c in np.argwhere(mism.any(axis=0)):
        v0 = int(conf_vis[r, c])
        for s, ln, v in _cell_value_episodes(vis[:, r, c], v0):
            episodes.append({
                "video": video, "side": side, "run_id": run_id,
                "row": int(r) + HIDDEN_ROWS, "col": int(c),
                "conf": v0, "cnn": v, "len": ln,
                "mtype": _mismatch_type(v0, v),
                "frame_start": recs[a + s].frame_idx,
                "frame_end": recs[a + s + ln - 1].frame_idx,
                "t_start": round(recs[a + s].t, 3),
                "t_end": round(recs[a + s + ln - 1].t, 3),
                "from_run_start": s,
                "to_run_end": run_len - (s + ln),
                "run_len": run_len,
            })
    return episodes


def _analyze_side(
    recs: list[_FrameRec], video: str, side: str, game_start_sec: float,
) -> tuple[_SideAcc, list[dict]]:
    """1 side 分: run 分割 → フレーム集計 + エピソード抽出。"""
    acc = _SideAcc()
    for rec in recs:
        acc.state_counts[rec.state] = acc.state_counts.get(rec.state, 0) + 1
    runs, short = _segment_stable_runs(recs, game_start_sec)
    acc.short_run_frames = short
    episodes: list[dict] = []
    for run_id, (a, b) in enumerate(runs):
        cnn_stack = np.stack([recs[i].cnn for i in range(a, b + 1)])
        conf = recs[a].confirmed
        assert conf is not None
        acc.n_runs += 1
        acc.run_frames += b - a + 1
        _analyze_run_frames(acc, cnn_stack, conf, b - a + 1)
        episodes.extend(_analyze_run_episodes(
            recs, a, b, video, side, run_id, cnn_stack, conf))
    return acc, episodes


# ============================
# エピソード統計
# ============================


def _episode_stats(episodes: list[dict]) -> dict:
    """持続長分布と票数 (3/8/18) との比較を計算する。"""
    if not episodes:
        return {"n_episodes": 0}
    lens = np.array([e["len"] for e in episodes])
    total_frames = int(lens.sum())
    stats: dict = {
        "n_episodes": len(episodes),
        "mismatch_frames_total": total_frames,
        "len_median": float(np.median(lens)),
        "len_p90": float(np.percentile(lens, 90)),
        "len_max": int(lens.max()),
        "episodes_len1_pct": round(100.0 * float((lens == 1).sum()) / len(lens), 2),
        "episodes_len_ge_corr_pct": round(
            100.0 * float((lens >= CORRELATED_MIN_LEN).sum()) / len(lens), 2),
    }
    # 「不一致フレームのうち、持続長 >= 票数 のエピソードに属する割合」
    # = その票数の多数決では原理的に消せない誤りフレームの割合
    for w in VOTE_WINDOWS:
        share = float(lens[lens >= w].sum()) / max(total_frames, 1)
        stats[f"frames_in_episodes_len_ge_{w}_pct"] = round(100.0 * share, 2)
    # 型別の持続長中央値
    by_type: dict = {}
    for e in episodes:
        by_type.setdefault(e["mtype"], []).append(e["len"])
    stats["by_type"] = {
        k: {"n": len(v), "len_median": float(np.median(v)),
            "len_max": int(max(v)), "frames": int(sum(v))}
        for k, v in sorted(by_type.items())
    }
    return stats


# ============================
# 切り抜き (pass 2)
# ============================


def _region_for_side(side: str) -> BoardRegion:
    return DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION


def _select_crop_episodes(episodes: list[dict]) -> list[dict]:
    """型ごとに持続長上位 CROPS_PER_TYPE 件を切り抜き対象に選ぶ。"""
    by_type: dict[str, list[dict]] = {}
    for e in episodes:
        by_type.setdefault(e["mtype"], []).append(e)
    out: list[dict] = []
    for _mt, eps in sorted(by_type.items()):
        eps_sorted = sorted(eps, key=lambda x: -x["len"])
        out.extend(eps_sorted[:CROPS_PER_TYPE])
    return out


def _crop_one(
    frame: np.ndarray, region: BoardRegion, row: int, col: int,
) -> np.ndarray:
    """対象セル周辺 (±CROP_MARGIN_CELLS) を切り抜き、対象セルを赤枠で示す。"""
    cw, ch = region.cell_width, region.cell_height
    vr = row - HIDDEN_ROWS
    x1 = int(region.x + (col - CROP_MARGIN_CELLS) * cw)
    y1 = int(region.y + (vr - CROP_MARGIN_CELLS) * ch)
    x2 = int(region.x + (col + 1 + CROP_MARGIN_CELLS) * cw)
    y2 = int(region.y + (vr + 1 + CROP_MARGIN_CELLS) * ch)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c = min(x2, frame.shape[1])
    y2c = min(y2, frame.shape[0])
    crop = frame[y1c:y2c, x1c:x2c].copy()
    # 対象セルの枠 (crop 座標系)
    rx1 = int(region.x + col * cw) - x1c
    ry1 = int(region.y + vr * ch) - y1c
    cv2.rectangle(crop, (rx1, ry1), (rx1 + int(cw), ry1 + int(ch)),
                  CROP_RECT_COLOR, 2)
    if CROP_SCALE != 1:
        crop = cv2.resize(crop, None, fx=CROP_SCALE, fy=CROP_SCALE,
                          interpolation=cv2.INTER_NEAREST)
    return crop


def _save_crops(stem: str, selected: list[dict], out_dir: Path) -> list[str]:
    """選定エピソードの中間フレームを動画から再取得し切り抜きを保存する。"""
    crop_dir = out_dir / "crops" / stem
    crop_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"video_{stem}.mp4"))
    saved: list[str] = []
    for e in sorted(selected, key=lambda x: x["frame_start"]):
        mid = (e["frame_start"] + e["frame_end"]) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(mid))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        crop = _crop_one(frame, _region_for_side(e["side"]), e["row"], e["col"])
        name = (f"{e['mtype']}_len{e['len']}_{e['side']}_r{e['row']}c{e['col']}"
                f"_conf{e['conf']}_cnn{e['cnn']}_f{mid}.png")
        cv2.imwrite(str(crop_dir / name), crop)
        saved.append(name)
    cap.release()
    return saved


# ============================
# 出力
# ============================


def _acc_summary(acc: _SideAcc) -> dict:
    """集計器を JSON 化可能な dict に変換する。"""
    total_mism = sum(acc.by_type.values())
    core_mism = sum(acc.core_by_type.values())
    return {
        "compared": acc.compared,
        "mismatch_total": total_mism,
        "mismatch_rate_pct": round(100.0 * total_mism / max(acc.compared, 1), 4),
        "core_compared": acc.core_compared,
        "core_mismatch_total": core_mism,
        "core_mismatch_rate_pct": round(
            100.0 * core_mism / max(acc.core_compared, 1), 4),
        "by_type": dict(sorted(acc.by_type.items())),
        "core_by_type": dict(sorted(acc.core_by_type.items())),
        "unknown_cnn": acc.unknown_cnn,
        "unknown_conf": acc.unknown_conf,
        "n_runs": acc.n_runs,
        "run_frames": acc.run_frames,
        "short_run_frames": acc.short_run_frames,
        "state_counts": acc.state_counts,
        "edge_mismatch": acc.edge_mismatch,
        "edge_compared": acc.edge_compared,
        "cell_mismatch_grid": acc.cell_mismatch.tolist(),
        "cell_compared_grid": acc.cell_compared.tolist(),
    }


def _write_episodes_csv(episodes: list[dict], path: Path) -> None:
    cols = ["video", "side", "run_id", "row", "col", "conf", "cnn", "len",
            "mtype", "frame_start", "frame_end", "t_start", "t_end",
            "from_run_start", "to_run_end", "run_len"]
    lines = [",".join(cols)]
    for e in episodes:
        lines.append(",".join(str(e[c]) for c in cols))
    path.write_text("\n".join(lines), encoding="utf-8")


def _process_target(target: str, out_dir: Path, smoke: bool) -> None:
    """1 ターゲット (stem:start:dur:game_start) を処理し結果を書き出す。"""
    stem, s_start, s_dur, s_gs = target.split(":")
    start_sec, dur_sec, game_start = float(s_start), float(s_dur), float(s_gs)
    if smoke:
        dur_sec = SMOKE_DUR_SEC
    _log(f"[{stem}] 走査開始 {start_sec:.1f}s + {dur_sec:.1f}s (試合開始 {game_start:.1f}s)")
    t0 = time.time()
    recs_1p, recs_2p, fps = _scan_video(stem, start_sec, dur_sec)
    _log(f"[{stem}] 走査完了 {time.time() - t0:.0f}s fps={fps:.1f}")
    all_eps: list[dict] = []
    summary: dict = {"video": stem, "fps": fps, "start_sec": start_sec,
                     "dur_sec": dur_sec, "game_start_sec": game_start,
                     "config": {"stable_frame_count": STABLE_FRAME_COUNT,
                                "temporal_smoothing": TEMPORAL_SMOOTHING,
                                "force_in_match": FORCE_IN_MATCH,
                                "min_run_frames": MIN_RUN_FRAMES,
                                "head_trim": HEAD_TRIM_FRAMES,
                                "tail_trim": TAIL_TRIM_FRAMES}}
    for side, recs in (("1P", recs_1p), ("2P", recs_2p)):
        acc, eps = _analyze_side(recs, stem, side, game_start)
        summary[side] = _acc_summary(acc)
        summary[side]["episode_stats"] = _episode_stats(eps)
        all_eps.extend(eps)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_episodes_csv(all_eps, out_dir / f"episodes_{stem}.csv")
    saved = _save_crops(stem, _select_crop_episodes(all_eps), out_dir)
    summary["crops_saved"] = saved
    (out_dir / f"summary_{stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"[{stem}] 完了: episodes={len(all_eps)} crops={len(saved)} → {out_dir}")


# ============================
# 集約モード
# ============================


def _aggregate(out_dir: Path) -> None:
    """summary_*.json を集約して全体表を表示する。"""
    files = sorted(out_dir.glob("summary_*.json"))
    if not files:
        _log(f"summary_*.json が見つかりません: {out_dir}")
        return
    rows: list[str] = ["video side fps compared mism rate% core_rate% "
                       "runs run_frames"]
    for f in files:
        s = json.loads(f.read_text(encoding="utf-8"))
        for side in ("1P", "2P"):
            a = s[side]
            rows.append(
                f"{s['video']} {side} {s['fps']:.0f} {a['compared']} "
                f"{a['mismatch_total']} {a['mismatch_rate_pct']} "
                f"{a['core_mismatch_rate_pct']} {a['n_runs']} {a['run_frames']}")
    text = "\n".join(rows)
    print(text)
    (out_dir / "summary_all.txt").write_text(text, encoding="utf-8")


def main() -> None:
    cv2.setNumThreads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS),
                    help="stem:start_sec:dur_sec:game_start_sec")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--smoke", action="store_true", help="先頭8秒のみの動作確認")
    ap.add_argument("--aggregate", action="store_true", help="集約表示のみ")
    args = ap.parse_args()
    if args.aggregate:
        _aggregate(args.out_dir)
        return
    targets = args.targets[:1] if args.smoke else args.targets
    for tgt in targets:
        _process_target(tgt, args.out_dir, args.smoke)


if __name__ == "__main__":
    main()
