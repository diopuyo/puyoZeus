"""設置確定レイテンシ A/B 実験 (2026-08-08、read-only診断)。

## 背景
userレビュー指摘: 盤面上部にツモを置いて即発火するケースで、設置の盤面反映が
間に合わず発火(連鎖開始)検知が失敗する。「確定を3フレーム程度に短縮できれば
検知できるはず」という仮説を検証する。

## 特定したパラメータ
`src/board_state_machine.py:107 STABLE_RECOVERY_MIN_FRAMES = 8`
(コンストラクタ引数名 `recovery_min_frames`)。

盤面確定への書き込みには2つのドアがある (docs/RECOGNITION_PLAIN_2026-08-05.md §3-2):
  - ドア1 (遷移の一括反映, `_merge_diff_only`): TSUMO_FALL→STABLE 等の
    state 遷移の瞬間に diff を即時反映。多フレーム待ちなし。
  - ドア2 (静止中のじわじわ修正, `_apply_stable_recovery_gate` /
    `_collect_recovery_candidates`): STABLE 中に CNN==HSV の合意が
    `recovery_min_frames` フレーム連続したセルのみ確定盤面を書き換える。

「上部に置いて即発火」ケースは、着地の可視時間があまりに短く
`TsumoPhaseDetector.consec_threshold`(=2) を満たせず TSUMO_FALL 状態に
遷移しないまま (= state は STABLE のまま) 連鎖演出に飲まれるケースが疑われる。
この場合、新規ぷよの書き込みはドア1を経由せずドア2 (方向1: 空→色) のみに
依存するため、`recovery_min_frames` の長さがそのまま「検知できるか否か」を
左右する。既存診断ツール `scripts/_diag_placement_confirm_frames_2026-07-25.py`
はこの引数を `--recovery-min-frames` として既にサポートしている
(board_state_machine.py 側コンストラクタ引数も既存、後方互換 = 省略時 None は
一切変更しない)。本スクリプトはこのツールを流用し、対象クリップ向けに
以下を追加計測する:
  A. 発火直前配置の反映成否 (CHAIN 遷移前に confirmed へ反映できたか)
  B. STABLE 中の確定盤面セル書き換え量 (汚染量の代理指標)
  C. 訂正レイテンシの層別 (設置前セルが「空で確定」か「誤値で確定」か)
  D. (C) の遅延セルについて、実パッチ HSV + 背景指紋 NCC を保存し
     残像説/ヒステリシス説を切り分ける

読み取り専用診断。src/ は一切変更しない。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

cv2.setNumThreads(1)

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_UNKNOWN, HIDDEN_ROWS  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from src.background_fingerprint import (  # noqa: E402
    CellPatchFingerprint, PatchBackgroundFingerprint, PATCH_NCC_EMPTY_THRESHOLD,
    _extract_cell_patch_hsv, CELL_SAMPLE_RATIO,
)

# 既存診断ツールをモジュールとして読み込む (ファイル名にハイフンがあるため
# 通常の import 文は使えず importlib で明示ロードする)。
_DIAG_PATH = PROJ_ROOT / "scripts" / "_diag_placement_confirm_frames_2026-07-25.py"
_spec = importlib.util.spec_from_file_location("_diag_pcf", _DIAG_PATH)
diag = importlib.util.module_from_spec(_spec)
sys.modules["_diag_pcf"] = diag  # dataclass の __module__ 解決に必要
_spec.loader.exec_module(diag)  # type: ignore[union-attr]

# ============================
# 定数
# ============================
CLIP_PATH: Path = PROJ_ROOT / "data" / "verify" / "youtube_demo_2026-08-07" / "dio_vs_ts_m01_clip.mp4"
OUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "recovery_min_frames_ab_2026-08-08"
PATCH_DIR: Path = PROJ_ROOT / "data" / "verify" / "correction_latency_patches_2026-08-08"

# A/B 3水準 (現行=STABLE_RECOVERY_MIN_FRAMES、5、3)
LEVELS: dict[str, int] = {"current_8": 8, "level_5": 5, "level_3": 3}

# 「置いてすぐ発火」候補と判定する、設置(place_idx)からCHAIN遷移までの許容秒数。
INSTANT_CHAIN_WINDOW_SEC: float = 3.0

# gate check 対象動画 (新標準構成、先頭10分)
GATE_VIDEOS: dict[str, str] = {"c10": "c10", "c22": "c22"}
GATE_MAX_SEC: float = 600.0

# phase_l_video_quality_gate.py の検査1相当の定数を踏襲。
GATE_OFFSET_LO_SEC: float = 1.0
GATE_OFFSET_HI_SEC: float = 3.0
GATE_ROW_LO: int = 1
GATE_ROW_HI_EXCL: int = 10
GATE_FAIL_RATE: float = 0.005

# 訂正遅延パッチ保存: 上限件数 (I/O 節約)
MAX_PATCH_EVENTS: int = 6
MAX_PATCH_FRAMES_PER_EVENT: int = 6


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================
# 「新標準構成」パイプライン構築 (2026-08-07 phase_l regen ジョブと同一設定)
# 出典: scripts/_gen_jobs_phase_l_regen_2026-08-07.py PHASE_L_REGEN_FLAGS
# ============================

# 新標準構成フラグ (第4機構修正A' 含む、2026-08-07 確定)。
# force_in_match は診断目的で False (診断側の既定) にする: collect_boards_lean.py
# は True 固定だが、それは npz 側の別経路 (_SharedGameCounter) で試合境界を
# 追う前提のため。本スクリプトは pipeline の is_match_active 自体で試合開始を
# 検出するので False (自然遷移) が正しい。それ以外は本番既定と同一。
STANDARD_CONFIG_KWARGS: dict = dict(
    stable_frame_count=3,
    load_score_ocr=True,
    enable_chain_tracker=True,
    load_next_detector=True,
    force_in_match=False,
    enable_effect_gate=True,
    enable_burst_guard_v2=True,
    enable_transition_merge_guard=True,
    burst_gate_open_threshold=0.954,
    enable_hidden_row_burst_guard=True,
    enable_match_transition_debounce=True,
)


def _build_pipeline(
    recovery_min_frames: "int | None", *, extra_kwargs: "dict | None" = None,
) -> "RecognitionPipeline":
    """新標準構成 + recovery_min_frames 上書き済み pipeline を構築する。

    recovery_min_frames=None なら一切上書きしない (本番 STABLE_RECOVERY_MIN_FRAMES
    =8 のまま、bit-identical)。
    extra_kwargs: 非対称案追試 (2026-08-08、コーディネーター追加依頼) 用。
    `enable_asymmetric_recovery_min_frames`/`recovery_add_min_frames` は
    load_default の正規 kwarg なので monkeypatch 不要でそのまま渡す。
    """
    kwargs = dict(STANDARD_CONFIG_KWARGS)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    pipe = RecognitionPipeline.load_default(**kwargs)
    if recovery_min_frames is not None:
        for attr in ("_sm_1p", "_sm_2p"):
            sm = getattr(pipe, attr, None)
            if sm is not None and hasattr(sm, "_recovery_min_frames"):
                sm._recovery_min_frames = max(1, int(recovery_min_frames))
    return pipe


def _collect_records_standard(
    video_path: Path, video_id: str, start_sec: float, max_sec: float,
    recovery_min_frames: "int | None", *, extra_kwargs: "dict | None" = None,
) -> tuple[list, list, float, "RecognitionPipeline"]:
    """新標準構成 pipeline で動画を走査し、1P/2P の _FrameRec 列を返す。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + max_sec) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipe = _build_pipeline(recovery_min_frames, extra_kwargs=extra_kwargs)
    pipe.set_video_id(video_id)

    recs_1p: list = []
    recs_2p: list = []
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
        for side_recs, side_res in ((recs_1p, r.p1), (recs_2p, r.p2)):
            side_recs.append(diag._FrameRec(
                frame_idx=fi, t=t, is_match_active=r.is_match_active,
                cnn_grid=side_res.cnn_board._grid.copy(),
                confirmed_grid=(
                    side_res.confirmed_board._grid.copy()
                    if side_res.confirmed_board is not None else None
                ),
                next_pair=side_res.next_pair,
                state=str(getattr(side_res, "state", "")).replace("BoardState.", ""),
            ))
        fi += 1
        n_read += 1
        if n_read % 1800 == 0:
            _log(f"[{video_id}] t={t:.1f}s まで処理済み ({n_read} frames)")
    cap.release()
    return recs_1p, recs_2p, fps, pipe


# ============================
# A. クリップ走行 (1 水準分)
# ============================


def _run_clip_level(level_name: str, recovery_min_frames: int, *, smoke: bool = False) -> dict:
    """クリップ全体を 1 水準で走査し、各種イベント・統計を返す。"""
    max_sec = 12.0 if smoke else 80.0
    recs_1p, recs_2p, fps, _pipe = _collect_records_standard(
        CLIP_PATH, "m01clip", 0.0, max_sec, recovery_min_frames,
    )
    return {"recs_1p": recs_1p, "recs_2p": recs_2p, "fps": fps, "level": level_name}


def _detect_chain_entries(records: list) -> list[int]:
    """state が CHAIN に遷移した frame index (records 内 index) 一覧。"""
    out: list[int] = []
    for i in range(1, len(records)):
        if records[i].state == "CHAIN" and records[i - 1].state != "CHAIN":
            out.append(i)
    return out


def _classify_baseline(
    records: list, cell_a: tuple, cell_b: tuple, place_idx: int,
) -> str:
    """設置直前の confirmed_board 値で "empty" / "misconfirmed" を分類する。

    place_idx の 1 frame 前の confirmed_grid を見る (無ければ "unknown")。
    片方でも非空 (色 or おじゃま) なら "misconfirmed" とする。
    """
    idx = place_idx - 1
    if idx < 0:
        return "unknown"
    cg = records[idx].confirmed_grid
    if cg is None:
        return "unknown"
    va, vb = int(cg[cell_a]), int(cg[cell_b])
    if va in (COLOR_EMPTY, COLOR_UNKNOWN) and vb in (COLOR_EMPTY, COLOR_UNKNOWN):
        return "empty"
    return "misconfirmed"


def _analyze_fire_detection(
    records: list, video: str, side: str, fps: float,
) -> dict:
    """発火直前配置の反映成否 + 訂正レイテンシ層別 (a/b) を計測する。"""
    events, meta = diag._build_placement_events(records, video, side, fps, 0.0)
    chain_entries = _detect_chain_entries(records)
    window_frames = round(fps * INSTANT_CHAIN_WINDOW_SEC)

    instant_chain_cases: list[dict] = []
    for e in events:
        place_idx = next(
            (i for i, r in enumerate(records) if r.frame_idx == e.frame_place), None,
        )
        if place_idx is None:
            continue
        nearest_chain = min(
            (ci for ci in chain_entries if 0 <= ci - place_idx <= window_frames),
            default=None,
        )
        if nearest_chain is None:
            continue
        reflect_idx = next(
            (i for i, r in enumerate(records) if r.frame_idx == e.frame_reflect),
            None,
        ) if e.frame_reflect is not None else None
        captured_before_chain = (
            reflect_idx is not None and reflect_idx <= nearest_chain
        )
        baseline_cls = _classify_baseline(records, e.cell_a, e.cell_b, place_idx)
        instant_chain_cases.append({
            "t_place": e.t_place, "t_chain": records[nearest_chain].t,
            "gap_sec": records[nearest_chain].t - e.t_place,
            "captured_before_chain": captured_before_chain,
            "delay_frames_total": e.delay_frames_total,
            "baseline_class": baseline_cls,
        })

    # 訂正レイテンシ層別 (全イベント、発火近接に限らない)
    strat_a, strat_b = [], []
    for e in events:
        place_idx = next(
            (i for i, r in enumerate(records) if r.frame_idx == e.frame_place), None,
        )
        if place_idx is None or e.delay_frames_total is None:
            continue
        cls = _classify_baseline(records, e.cell_a, e.cell_b, place_idx)
        if cls == "empty":
            strat_a.append(e.delay_frames_total)
        elif cls == "misconfirmed":
            strat_b.append(e.delay_frames_total)

    return {
        "n_events": len(events),
        "n_instant_chain_cases": len(instant_chain_cases),
        "n_captured_before_chain": sum(
            1 for c in instant_chain_cases if c["captured_before_chain"]
        ),
        "instant_chain_cases": instant_chain_cases,
        "strat_empty_delay_median": (
            float(np.median(strat_a)) if strat_a else None
        ),
        "strat_empty_n": len(strat_a),
        "strat_misconfirmed_delay_median": (
            float(np.median(strat_b)) if strat_b else None
        ),
        "strat_misconfirmed_n": len(strat_b),
        "strat_empty_delays": strat_a,
        "strat_misconfirmed_delays": strat_b,
    }


def _measure_pollution(records: list) -> dict:
    """STABLE 中の確定盤面セル書き換え量 (汚染量の代理指標)。

    frame間で confirmed_grid が変化したセルの総数をカウントする
    (試合非アクティブ・NON-STABLE 中は対象外)。
    """
    total_flips = 0
    n_stable_frames = 0
    prev = None
    for r in records:
        if not r.is_match_active or r.state not in ("STABLE",):
            prev = None
            continue
        n_stable_frames += 1
        if prev is not None and r.confirmed_grid is not None and prev is not None:
            total_flips += int(np.sum(r.confirmed_grid != prev))
        prev = r.confirmed_grid
    return {"total_cell_flips": total_flips, "n_stable_frames": n_stable_frames}


# ============================
# B. ゲート検査 (c10 / c22 先頭10分、試合開始直後空盤面)
# ============================


def _run_gate_check(video_stem: str, recovery_min_frames: int, *, max_sec: float) -> dict:
    video_path = PROJ_ROOT / "data" / "frames" / f"video_{video_stem}.mp4"
    recs_1p, recs_2p, fps, _pipe = _collect_records_standard(
        video_path, video_stem, 0.0, max_sec, recovery_min_frames,
    )
    result = {}
    for side_name, records in (("1P", recs_1p), ("2P", recs_2p)):
        starts = [
            i for i in range(1, len(records))
            if records[i].is_match_active and not records[i - 1].is_match_active
        ]
        nonempty_total = 0
        cell_total = 0
        for s in starts:
            t0 = records[s].t
            lo, hi = t0 + GATE_OFFSET_LO_SEC, t0 + GATE_OFFSET_HI_SEC
            for r in records[s:]:
                if r.t < lo:
                    continue
                if r.t > hi:
                    break
                cg = r.confirmed_grid
                if cg is None:
                    continue
                window = cg[GATE_ROW_LO:GATE_ROW_HI_EXCL, :]
                nonempty_total += int(np.sum(window != COLOR_EMPTY))
                cell_total += window.size
        rate = (nonempty_total / cell_total) if cell_total else None
        result[side_name] = {
            "n_match_starts": len(starts), "nonempty_rate": rate,
            "cell_total": cell_total, "fail": (rate is not None and rate > GATE_FAIL_RATE),
        }
    return result


# ============================
# C. パッチ抽出 (残像 vs ヒステリシス切り分け + 背景指紋NCC)
# ============================


def _cell_visible_row(row: int) -> int:
    return row - HIDDEN_ROWS


def _save_patches_for_case(
    pipe, video_path: Path, side: str, cell: tuple[int, int],
    place_frame_idx: int, reflect_frame_idx: "int | None", fps: float,
    out_dir: Path, tag: str,
) -> list[dict]:
    """1 セルにつき、設置〜反映(または追跡上限)までの数フレームの実パッチを保存し、
    HSV 中央値・背景指紋 NCC を記録する。
    """
    region = pipe._reader._p1_region if side == "1P" else pipe._reader._p2_region
    bg_fp = pipe._reader._bg_fp_for_region(region)
    row, col = cell
    vrow = _cell_visible_row(row)
    end_idx = (
        reflect_frame_idx if reflect_frame_idx is not None
        else place_frame_idx + round(fps * 2.0)
    )
    frame_ids = np.linspace(
        place_frame_idx, end_idx,
        num=min(MAX_PATCH_FRAMES_PER_EVENT, max(2, end_idx - place_frame_idx + 1)),
        dtype=int,
    )
    frame_ids = sorted(set(int(f) for f in frame_ids))

    cap = cv2.VideoCapture(str(video_path))
    out_records: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for fi in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        patch_hsv = _extract_cell_patch_hsv(
            hsv, region.x, region.y, region.width, region.height,
            vrow, col, CELL_SAMPLE_RATIO,
        )
        h_med = float(np.median(patch_hsv[:, :, 0]))
        s_med = float(np.median(patch_hsv[:, :, 1]))
        v_med = float(np.median(patch_hsv[:, :, 2]))
        ncc = None
        if bg_fp is not None and hasattr(bg_fp, "cell_at_patch"):
            cur_fp = CellPatchFingerprint(patch_hsv=patch_hsv)
            bg_cell = bg_fp.cell_at_patch(vrow, col)
            ncc = float(cur_fp.ncc_to(bg_cell))
        # BGR crop も保存 (目視用)
        x1, y1, x2, y2 = region.cell_sample_rect(row, col)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        fname = f"{tag}_r{row}c{col}_f{fi}.png"
        if crop.size > 0:
            cv2.imwrite(str(out_dir / fname), crop)
        out_records.append({
            "frame_idx": fi, "t": fi / fps, "h_med": h_med, "s_med": s_med,
            "v_med": v_med, "bg_fp_ncc": ncc,
            "bg_fp_is_empty_by_ncc": (
                (ncc >= PATCH_NCC_EMPTY_THRESHOLD) if ncc is not None else None
            ),
            "patch_file": fname,
        })
    cap.release()
    return out_records


def _run_patch_analysis(
    pipe, records: list, video_path: Path, side: str, fps: float,
) -> list[dict]:
    """訂正レイテンシが大きい (misconfirmed 層) イベントを抽出し、パッチを保存する。"""
    events, _ = diag._build_placement_events(records, "m01clip", side, fps, 0.0)
    candidates = []
    for e in events:
        place_idx = next(
            (i for i, r in enumerate(records) if r.frame_idx == e.frame_place), None,
        )
        if place_idx is None or e.delay_frames_total is None:
            continue
        cls = _classify_baseline(records, e.cell_a, e.cell_b, place_idx)
        if cls == "misconfirmed" and e.delay_frames_total >= 3:
            candidates.append((e, place_idx))
    candidates.sort(key=lambda pair: -(pair[0].delay_frames_total or 0))
    candidates = candidates[:MAX_PATCH_EVENTS]

    out: list[dict] = []
    for e, place_idx in candidates:
        reflect_idx = None
        if e.frame_reflect is not None:
            reflect_idx = next(
                (i for i, r in enumerate(records) if r.frame_idx == e.frame_reflect),
                None,
            )
        for cell, tag in ((e.cell_a, "a"), (e.cell_b, "b")):
            patches = _save_patches_for_case(
                pipe, video_path, side, cell, e.frame_place,
                (records[reflect_idx].frame_idx if reflect_idx is not None else None),
                fps, PATCH_DIR, f"{side}_t{e.t_place:.1f}_{tag}",
            )
            out.append({
                "video": "m01clip", "side": side, "t_place": e.t_place,
                "cell": cell, "delay_frames_total": e.delay_frames_total,
                "truth_color": (
                    e.truth_color_a if tag == "a" else e.truth_color_b
                ),
                "patches": patches,
            })
    return out


# ============================
# main
# ============================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-gate", action="store_true", help="c10/c22 ゲート検査を省略する")
    ap.add_argument("--skip-patches", action="store_true", help="パッチ抽出を省略する")
    ap.add_argument(
        "--gate-max-sec", type=float, default=GATE_MAX_SEC,
        help="ゲート検査の走査秒数 (既定 600 = 10分)。時間短縮したい場合に指定。",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict = {"levels": {}}

    # --- A. クリップ 3水準 ---
    for level_name, rmf in LEVELS.items():
        _log(f"[clip] level={level_name} (recovery_min_frames={rmf}) 走査開始")
        t0 = time.time()
        run = _run_clip_level(level_name, rmf, smoke=args.smoke)
        _log(f"[clip] level={level_name} 走査完了 ({time.time() - t0:.1f}s)")
        level_result: dict = {}
        for side, records in (("1P", run["recs_1p"]), ("2P", run["recs_2p"])):
            fire = _analyze_fire_detection(records, "m01clip", side, run["fps"])
            pollution = _measure_pollution(records)
            level_result[side] = {"fire": fire, "pollution": pollution}
        result["levels"][level_name] = level_result
        _log(
            f"[clip] level={level_name}: "
            f"1P instant_chain={level_result['1P']['fire']['n_instant_chain_cases']} "
            f"captured={level_result['1P']['fire']['n_captured_before_chain']} / "
            f"2P instant_chain={level_result['2P']['fire']['n_instant_chain_cases']} "
            f"captured={level_result['2P']['fire']['n_captured_before_chain']}",
        )

    # --- B. c10/c22 ゲート検査 ---
    if not args.skip_gate:
        gate_result: dict = {}
        gate_sec = 30.0 if args.smoke else args.gate_max_sec
        for video_stem in GATE_VIDEOS.values():
            gate_result[video_stem] = {}
            for level_name, rmf in LEVELS.items():
                _log(f"[gate] video={video_stem} level={level_name} 走査開始 ({gate_sec:.0f}s)")
                t0 = time.time()
                gate_result[video_stem][level_name] = _run_gate_check(
                    video_stem, rmf, max_sec=gate_sec,
                )
                _log(f"[gate] video={video_stem} level={level_name} 完了 ({time.time() - t0:.1f}s)")
        result["gate"] = gate_result

    # --- C. パッチ抽出 (current_8 水準のみ、misconfirmed 層のワースト事例) ---
    if not args.skip_patches:
        _log("[patch] current_8 水準を再走行してパッチ抽出用の pipeline を保持")
        recs_1p, recs_2p, fps, pipe = _collect_records_standard(
            CLIP_PATH, "m01clip", 0.0, 12.0 if args.smoke else 80.0,
            LEVELS["current_8"],
        )
        patch_result = []
        for side, records in (("1P", recs_1p), ("2P", recs_2p)):
            patch_result += _run_patch_analysis(pipe, records, CLIP_PATH, side, fps)
        result["patch_analysis"] = patch_result
        _log(f"[patch] {len(patch_result)} 件のセルパッチを {PATCH_DIR} に保存")

    out_path = OUT_DIR / "result.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    _log(f"[DONE] 出力: {out_path}")


if __name__ == "__main__":
    main()
