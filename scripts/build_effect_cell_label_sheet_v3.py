"""エフェクト有無 セルラベルシート 第3弾 準備スクリプト (2026-08-04)。

## 背景 (memory project_effect_gate_v1_failure_2026-08-03 / 較正2026-08-04)
第1弾ラベル (scripts/build_effect_cell_label_sheet.py、36枚、機械推定窓) の
較正結果 (data/verify/effect_detector_calibration_2026-08-04/calibration_report.md)
で、窓レベル同時性判定 (v_mean、row1-3内15セル以上同時該当) がフレームAUC 0.978
と有望と判明したが、
  1. n=36・真陽性7件は検出力不足 (最低100フレーム規模が必要)
  2. 「連鎖数テロップ (自分の連鎖の「Xれんさ!」表示) の発光」が予告おじゃま
     バーストと混同する新失敗モードが判明 (唯一の反例 c21 t=2764.13)
という2つの課題が残った。本スクリプトは次段として、
  - 機械推定窓でなく **発火イベント台帳の実イベント** から窓を取る (窓の質向上)
  - **テロップ負例を意図的に含める** (失敗モードを正面から測る)
100フレーム規模のラベル候補を用意する (ラベル付け自体は行わない)。

## 第1弾との差分
- 窓ソース: study_effect_signature (機械推定・連鎖規模ビン単位の中間点サンプル)
  ではなく data/indicators_v2/exchange_labels_regen_synth79_2026-08-04.csv の
  実イベント (t_sec=連鎖終了、is_synthetic_terminal_event=0 のみ) を直接使う。
- 層構成: burst(40) / smoke(25) / telop_negative(15) / zenkeshi(5) / baseline(15)
  の5層 (第1弾は burst/smoke/baseline の3層のみ、テロップ層が新規)。
- 動画プール: 79動画 (exchange_labels_regen_synth79 に含まれる全動画、
  boards_lean_regen_2026-07-31 npz + ローカルmp4キャッシュ両方の存在を確認済み)。

## 出力 (data/verify/effect_cell_label_v3_2026-08-04/)
    frames/<video>_t<t_sec>_<side>_<layer>_full.png        実画面フルフレーム
    frames/<video>_t<t_sec>_<side>_<layer>_board_crop.png  盤面クロップ(可視12行)
    labeling_sheet.csv     候補一覧 (label_tool_v3.htmlの入力)
    label_sheet.md         説明書き付き一覧 (Windowsパスリンク)

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.build_effect_cell_label_sheet_v3
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.build_full_board_label_tool import crop_visible_board_region  # noqa: E402
from scripts.extract_exchange_event_frames import (  # noqa: E402
    grab_frame, resolve_cached_video_path, to_windows_path,
)
from src.board import (  # noqa: E402
    BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, HIDDEN_ROWS,
)
from src.indicators_v2 import CHAIN_ANIM_PER_STEP_SEC  # noqa: E402

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

OUTPUT_DIR: Path = Path("data/verify/effect_cell_label_v3_2026-08-04")
FRAMES_SUBDIR_NAME: str = "frames"

FIRE_EVENTS_CSV: Path = (
    Path("data/indicators_v2") / "exchange_labels_regen_synth79_2026-08-04.csv"
)
NPZ_DIR: Path = Path("data/indicators_v2/boards_lean_regen_2026-07-31")

RANDOM_SEED: int = 20260804

# --- 層別 目標件数 (合計100) ---
BURST_TARGET_TOTAL: int = 40
SMOKE_TARGET_TOTAL: int = 25
TELOP_NEGATIVE_TARGET_TOTAL: int = 15
ZENKESHI_TARGET_TOTAL: int = 5
BASELINE_TARGET_TOTAL: int = 15

# 動画あたり上限 (全レイヤー合算の共有カウンタで適用、20動画以上への分散を保証:
# 100件 / MAX_CANDIDATES_PER_VIDEO=5 = 最低20動画)
MAX_CANDIDATES_PER_VIDEO: int = 5

# 生プールをオーバーサンプルしてから動画分散選定する倍率目安
_OVERSAMPLE_FACTOR: int = 3
BURST_OVERSAMPLE_TOTAL: int = BURST_TARGET_TOTAL * _OVERSAMPLE_FACTOR
SMOKE_OVERSAMPLE_TOTAL: int = SMOKE_TARGET_TOTAL * _OVERSAMPLE_FACTOR
TELOP_NEGATIVE_OVERSAMPLE_TOTAL: int = TELOP_NEGATIVE_TARGET_TOTAL * _OVERSAMPLE_FACTOR
BASELINE_OVERSAMPLE_TOTAL: int = BASELINE_TARGET_TOTAL * _OVERSAMPLE_FACTOR

# バースト窓 (予告おじゃま送付、受け側盤面): [t_sec-連鎖数*0.4-1, t_sec+1]
BURST_WINDOW_PRE_MARGIN_SEC: float = 1.0
BURST_WINDOW_POST_MARGIN_SEC: float = 1.0

# テロップ負例窓 (自分の連鎖中、自盤面): [t_sec-連鎖数*0.4, t_sec]
# (post marginは付けない。連鎖終了直後は既に静止しテロップも消えている想定のため)

# 煙窓 (おじゃま着弾、自盤面): 実増加時刻 ±1秒
SMOKE_WINDOW_MARGIN_SEC: float = 1.0
# おじゃま増加イベント検出のガード (study_effect_signature_2026-08-03 と同じ考え方)
SMOKE_MAX_GAP_SEC: float = 15.0
SMOKE_MIN_OJAMA_DELTA: int = 1
SMOKE_MAX_OJAMA_DELTA: int = 36  # 可視領域72セルの半分、異常値除外用

# 全消しテロップ: 全消し検出直後、テロップ描画待ちのオフセット (秒)
ZENKESHI_SAMPLE_OFFSET_SEC: float = 0.3

# 対照群 (平穏): 発火/おじゃま増加/全消しイベントの前後この秒数以内は除外
BASELINE_EXCLUSION_SEC: float = 5.0

LAYER_BURST: str = "burst"
LAYER_SMOKE: str = "smoke"
LAYER_TELOP_NEGATIVE: str = "telop_negative"
LAYER_ZENKESHI: str = "zenkeshi"
LAYER_BASELINE: str = "baseline"
LAYER_LABEL_JA: dict[str, str] = {
    LAYER_BURST: "バースト(予告おじゃま送付エフェクト疑い)",
    LAYER_SMOKE: "煙(おじゃま着弾エフェクト疑い)",
    LAYER_TELOP_NEGATIVE: "連鎖数テロップ負例(自分の連鎖中、混同注意)",
    LAYER_ZENKESHI: "全消しテロップ",
    LAYER_BASELINE: "対照(平穏、エフェクト無しの想定)",
}

CSV_HEADER: tuple[str, ...] = (
    "video_id", "t_sec", "side", "layer", "note",
    "image_full_frame", "image_board_crop",
)


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class EffectFrameCandidateV3:
    """1フレーム分のエフェクトラベル候補 (video_stem+side+t_secで一意)。"""

    video_stem: str
    side: str
    t_sec: float
    layer: str
    note: str = ""  # 補足 (連鎖数等、CSV/md表示用の自由記述)

    @property
    def video_id(self) -> str:
        """"c18" -> "video_c18" (既存ツール群の video_id 表記に合わせる)。"""
        return f"video_{self.video_stem}"


@dataclass
class _NpzSideIndex:
    """1 (video, side) 分の t_sec 昇順ソート済みインデックス。"""

    t_secs: "np.ndarray"
    grids: "np.ndarray"
    game_idxs: "np.ndarray"


# =============================================================================
# 1. 動画プール + npz アクセス
# =============================================================================


def load_video_pool(fire_csv_path: Path) -> list[str]:
    """発火イベント台帳から動画stem一覧 (tier filter済み既存資産) を取得する。"""
    df = pd.read_csv(fire_csv_path, usecols=["video_id"])
    stems = sorted({str(v).replace("video_", "") for v in df["video_id"].unique()})
    return stems


def load_npz_side_index(stem: str, side: str) -> "_NpzSideIndex | None":
    """1 (video, side) の t_sec 昇順ソート済み grids/game_idx を返す (無ければNone)。"""
    npz_path = NPZ_DIR / f"{stem}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    mask = data["side"] == side
    if not mask.any():
        return None
    order = np.argsort(data["t_sec"][mask])
    return _NpzSideIndex(
        t_secs=data["t_sec"][mask][order].astype(np.float64),
        grids=data["grids"][mask][order],
        game_idxs=data["game_idx"][mask][order].astype(np.int64),
    )


def load_fire_events(fire_csv_path: Path, video_stems: list[str]) -> pd.DataFrame:
    """実イベントのみ (synthetic除外) + 対象動画に絞った発火イベント台帳を返す。"""
    df = pd.read_csv(fire_csv_path)
    target_ids = {f"video_{s}" for s in video_stems}
    df = df[df["video_id"].isin(target_ids)].copy()
    df = df[df["is_synthetic_terminal_event"] == 0].copy()
    df = df.dropna(subset=["t_sec", "approx_fire_chains", "fire_side"]).copy()
    return df.reset_index(drop=True)


# =============================================================================
# 2. npz由来の実イベント検出 (おじゃま増加 / 全消し)
# =============================================================================


def find_ojama_increase_events(idx: "_NpzSideIndex") -> list[float]:
    """おじゃま実増加イベントの「増加後」スナップショット時刻一覧を返す。

    study_effect_signature_2026-08-03.find_ojama_increase_events と同じ判定
    条件 (同一game_idx・ギャップ上限・増加量の範囲) だが、窓の基準時刻は
    前後スナップショットの中間点でなく「増加後の実時刻」を採用する
    (タスク指定「実時刻の前後1秒」に対応するため)。
    """
    visible = idx.grids[:, HIDDEN_ROWS:BOARD_ROWS, :]
    col_counts = np.sum(visible == COLOR_OJAMA, axis=1)
    total_counts = np.sum(col_counts, axis=1)
    t_afters: list[float] = []
    for i in range(1, len(idx.t_secs)):
        delta = int(total_counts[i] - total_counts[i - 1])
        gap = float(idx.t_secs[i] - idx.t_secs[i - 1])
        same_game = idx.game_idxs[i] == idx.game_idxs[i - 1]
        if not same_game or gap > SMOKE_MAX_GAP_SEC:
            continue
        if not (SMOKE_MIN_OJAMA_DELTA <= delta <= SMOKE_MAX_OJAMA_DELTA):
            continue
        t_afters.append(float(idx.t_secs[i]))
    return t_afters


def find_zenkeshi_events(idx: "_NpzSideIndex") -> list[float]:
    """全消し (可視領域が全て空、おじゃま含めて全空) への遷移時刻一覧を返す。

    reference_puyo_rules_confirmed_2026-07-22 (全消し=おじゃま含め全空) に基づき、
    可視12行が全て COLOR_EMPTY のスナップショットで、直前スナップショットは
    全空でなかった (=遷移の瞬間) ものだけを採用する。
    """
    visible = idx.grids[:, HIDDEN_ROWS:BOARD_ROWS, :]
    is_all_empty = np.all(visible == COLOR_EMPTY, axis=(1, 2))
    events: list[float] = []
    for i in range(1, len(idx.t_secs)):
        same_game = idx.game_idxs[i] == idx.game_idxs[i - 1]
        if same_game and is_all_empty[i] and not is_all_empty[i - 1]:
            events.append(float(idx.t_secs[i]))
    return events


# =============================================================================
# 3. 層別 候補プール収集 (オーバーサンプル、動画分散選定は後段で行う)
# =============================================================================

_OPPOSITE_SIDE: dict[str, str] = {"1P": "2P", "2P": "1P"}


def collect_burst_pool(
    fire_df: pd.DataFrame, rng: np.random.Generator, n_oversample: int,
) -> list[EffectFrameCandidateV3]:
    """予告おじゃまバースト窓 (受け側=fire_sideの相手盤面) をオーバーサンプルする。

    窓 = [t_sec - approx_fire_chains*CHAIN_ANIM_PER_STEP_SEC - 1, t_sec + 1]
    (タスク指定の式、t_sec=連鎖終了時点)。
    """
    pool = fire_df.sample(
        n=min(n_oversample, len(fire_df)), random_state=int(rng.integers(0, 2**31 - 1)),
    )
    out: list[EffectFrameCandidateV3] = []
    for _, ev in pool.iterrows():
        stem = str(ev["video_id"]).replace("video_", "")
        opp_side = _OPPOSITE_SIDE[str(ev["fire_side"])]
        chain_len = float(ev["approx_fire_chains"])
        t_sec = float(ev["t_sec"])
        lo = max(0.0, t_sec - chain_len * CHAIN_ANIM_PER_STEP_SEC - BURST_WINDOW_PRE_MARGIN_SEC)
        hi = t_sec + BURST_WINDOW_POST_MARGIN_SEC
        sample_t = float(rng.uniform(lo, hi))
        out.append(EffectFrameCandidateV3(
            video_stem=stem, side=opp_side, t_sec=sample_t, layer=LAYER_BURST,
            note=f"相手連鎖{chain_len:.0f}連鎖の受け側",
        ))
    return out


def collect_telop_negative_pool(
    fire_df: pd.DataFrame, rng: np.random.Generator, n_oversample: int,
) -> list[EffectFrameCandidateV3]:
    """連鎖数テロップ負例 (発火した側自身の盤面、連鎖中) をオーバーサンプルする。

    窓 = [t_sec - approx_fire_chains*CHAIN_ANIM_PER_STEP_SEC, t_sec]
    (自分の連鎖アニメーション中のみ、失敗モード再現のため post margin は付けない)。
    """
    pool = fire_df.sample(
        n=min(n_oversample, len(fire_df)), random_state=int(rng.integers(0, 2**31 - 1)),
    )
    out: list[EffectFrameCandidateV3] = []
    for _, ev in pool.iterrows():
        stem = str(ev["video_id"]).replace("video_", "")
        side = str(ev["fire_side"])
        chain_len = float(ev["approx_fire_chains"])
        t_sec = float(ev["t_sec"])
        lo = max(0.0, t_sec - chain_len * CHAIN_ANIM_PER_STEP_SEC)
        sample_t = float(rng.uniform(lo, t_sec)) if lo < t_sec else t_sec
        out.append(EffectFrameCandidateV3(
            video_stem=stem, side=side, t_sec=sample_t, layer=LAYER_TELOP_NEGATIVE,
            note=f"自分の{chain_len:.0f}連鎖中(連鎖数テロップ表示中の想定)",
        ))
    return out


def collect_smoke_pool(
    video_stems: list[str], rng: np.random.Generator, n_oversample: int,
) -> list[EffectFrameCandidateV3]:
    """おじゃま実増加イベント (実時刻±1秒) をオーバーサンプルする (自盤面)。"""
    candidates: list[tuple[str, str, float]] = []
    for stem in video_stems:
        for side in ("1P", "2P"):
            idx = load_npz_side_index(stem, side)
            if idx is None:
                continue
            for t_after in find_ojama_increase_events(idx):
                candidates.append((stem, side, t_after))
    order = rng.permutation(len(candidates))[:n_oversample]
    out: list[EffectFrameCandidateV3] = []
    for i in order:
        stem, side, t_after = candidates[int(i)]
        lo = max(0.0, t_after - SMOKE_WINDOW_MARGIN_SEC)
        hi = t_after + SMOKE_WINDOW_MARGIN_SEC
        sample_t = float(rng.uniform(lo, hi))
        out.append(EffectFrameCandidateV3(
            video_stem=stem, side=side, t_sec=sample_t, layer=LAYER_SMOKE,
            note="おじゃま実増加±1秒",
        ))
    return out


def collect_zenkeshi_pool(
    video_stems: list[str], rng: np.random.Generator, n_oversample: int,
) -> list[EffectFrameCandidateV3]:
    """全消しテロップ遷移直後をオーバーサンプルする (自盤面)。"""
    candidates: list[tuple[str, str, float]] = []
    for stem in video_stems:
        for side in ("1P", "2P"):
            idx = load_npz_side_index(stem, side)
            if idx is None:
                continue
            for t_zenkeshi in find_zenkeshi_events(idx):
                candidates.append((stem, side, t_zenkeshi))
    order = rng.permutation(len(candidates))[:n_oversample]
    out: list[EffectFrameCandidateV3] = []
    for i in order:
        stem, side, t_zenkeshi = candidates[int(i)]
        sample_t = t_zenkeshi + ZENKESHI_SAMPLE_OFFSET_SEC
        out.append(EffectFrameCandidateV3(
            video_stem=stem, side=side, t_sec=sample_t, layer=LAYER_ZENKESHI,
            note="全消し遷移直後",
        ))
    return out


def _unsafe_intervals_for_video(
    stem: str, fire_df: pd.DataFrame, smoke_pool: list[EffectFrameCandidateV3],
    zenkeshi_pool: list[EffectFrameCandidateV3],
) -> list[tuple[float, float]]:
    """この動画で「平穏でない」時間帯 (発火/煙/全消しイベントの周辺) を返す。"""
    video_id = f"video_{stem}"
    times = list(fire_df.loc[fire_df["video_id"] == video_id, "t_sec"].to_numpy())
    times += [c.t_sec for c in smoke_pool if c.video_stem == stem]
    times += [c.t_sec for c in zenkeshi_pool if c.video_stem == stem]
    return [(t - BASELINE_EXCLUSION_SEC, t + BASELINE_EXCLUSION_SEC) for t in times]


def collect_baseline_pool(
    video_stems: list[str], fire_df: pd.DataFrame,
    smoke_pool: list[EffectFrameCandidateV3], zenkeshi_pool: list[EffectFrameCandidateV3],
    rng: np.random.Generator, n_oversample: int,
) -> list[EffectFrameCandidateV3]:
    """発火/煙/全消しから十分離れたSTABLE平穏スナップショットをオーバーサンプルする。"""
    candidates: list[tuple[str, str, float]] = []
    for stem in video_stems:
        unsafe = _unsafe_intervals_for_video(stem, fire_df, smoke_pool, zenkeshi_pool)
        for side in ("1P", "2P"):
            idx = load_npz_side_index(stem, side)
            if idx is None:
                continue
            for t_sec in idx.t_secs:
                if not any(lo <= t_sec <= hi for lo, hi in unsafe):
                    candidates.append((stem, side, float(t_sec)))
    order = rng.permutation(len(candidates))[:n_oversample]
    return [
        EffectFrameCandidateV3(
            video_stem=candidates[int(i)][0], side=candidates[int(i)][1],
            t_sec=candidates[int(i)][2], layer=LAYER_BASELINE, note="平穏対照",
        )
        for i in order
    ]


# =============================================================================
# 4. ラウンドロビン選定 (動画あたり上限を守りつつ偏りを抑える)
# =============================================================================


def round_robin_select(
    pool: list[EffectFrameCandidateV3], n_want: int, max_per_video: int,
    usage: dict[str, int],
) -> list[EffectFrameCandidateV3]:
    """動画あたり上限を守りながらプールから n_want 件をラウンドロビンで選ぶ。

    usage は全レイヤー共有のカウンタ (呼び出し元で使い回すことで、レイヤーを
    跨いだ動画あたり合計件数を制御し、少数動画への集中を防ぐ)。
    """
    picked: list[EffectFrameCandidateV3] = []
    remaining = list(pool)
    while len(picked) < n_want and remaining:
        used_this_round: set[str] = set()
        next_remaining: list[EffectFrameCandidateV3] = []
        for c in remaining:
            if len(picked) >= n_want:
                next_remaining.append(c)
                continue
            if c.video_stem in used_this_round or usage.get(c.video_stem, 0) >= max_per_video:
                next_remaining.append(c)
                continue
            picked.append(c)
            used_this_round.add(c.video_stem)
            usage[c.video_stem] = usage.get(c.video_stem, 0) + 1
        if not used_this_round:
            break  # これ以上ラウンドロビンで拾える候補がない
        remaining = next_remaining
    return picked


def collect_and_select_candidates(rng_seed: int = RANDOM_SEED) -> list[EffectFrameCandidateV3]:
    """5層それぞれオーバーサンプル -> 動画分散を保証しつつ100フレーム程度を選定する。"""
    rng = np.random.default_rng(rng_seed)
    video_stems = load_video_pool(FIRE_EVENTS_CSV)
    fire_df = load_fire_events(FIRE_EVENTS_CSV, video_stems)

    burst_pool = collect_burst_pool(fire_df, rng, BURST_OVERSAMPLE_TOTAL)
    telop_pool = collect_telop_negative_pool(fire_df, rng, TELOP_NEGATIVE_OVERSAMPLE_TOTAL)
    smoke_pool = collect_smoke_pool(video_stems, rng, SMOKE_OVERSAMPLE_TOTAL)
    zenkeshi_pool = collect_zenkeshi_pool(video_stems, rng, ZENKESHI_TARGET_TOTAL * _OVERSAMPLE_FACTOR)
    baseline_pool = collect_baseline_pool(
        video_stems, fire_df, smoke_pool, zenkeshi_pool, rng, BASELINE_OVERSAMPLE_TOTAL,
    )

    usage: dict[str, int] = {}
    selected: list[EffectFrameCandidateV3] = []
    selected += round_robin_select(burst_pool, BURST_TARGET_TOTAL, MAX_CANDIDATES_PER_VIDEO, usage)
    selected += round_robin_select(smoke_pool, SMOKE_TARGET_TOTAL, MAX_CANDIDATES_PER_VIDEO, usage)
    selected += round_robin_select(
        telop_pool, TELOP_NEGATIVE_TARGET_TOTAL, MAX_CANDIDATES_PER_VIDEO, usage,
    )
    zenkeshi_selected = round_robin_select(
        zenkeshi_pool, ZENKESHI_TARGET_TOTAL, MAX_CANDIDATES_PER_VIDEO, usage,
    )
    selected += zenkeshi_selected
    # 全消しは母集団に実イベントが無い場合がある (=でっち上げ禁止)。不足分は
    # 目標合計100を保つため平穏対照 (baseline) の枠に振り替える。
    baseline_target = BASELINE_TARGET_TOTAL + (ZENKESHI_TARGET_TOTAL - len(zenkeshi_selected))
    selected += round_robin_select(baseline_pool, baseline_target, MAX_CANDIDATES_PER_VIDEO, usage)
    return selected


# =============================================================================
# 5. 画像生成
# =============================================================================


def _frame_basename(c: EffectFrameCandidateV3) -> str:
    """ファイル名の共通部分 (video/t_sec/side/layerを含み衝突しないようにする)。"""
    return f"{c.video_stem}_t{c.t_sec:.2f}_{c.side}_{c.layer}"


def save_candidate_images(
    c: EffectFrameCandidateV3, frames_dir: Path,
) -> tuple["Path | None", "Path | None"]:
    """1候補分の実画面フルフレームPNG + 盤面クロップPNGを保存する (失敗時はNone)。"""
    video_path = resolve_cached_video_path(c.video_id)
    if video_path is None:
        return None, None
    frame = grab_frame(video_path, c.t_sec)
    if frame is None:
        return None, None
    base = _frame_basename(c)
    full_path = frames_dir / f"{base}_full.png"
    crop_path = frames_dir / f"{base}_board_crop.png"
    cv2.imwrite(str(full_path), frame)
    cv2.imwrite(str(crop_path), crop_visible_board_region(frame, c.side))
    return full_path, crop_path


# =============================================================================
# 6. 出力ファイル生成
# =============================================================================


def write_labeling_csv(rows: list[dict], out_path: Path) -> None:
    """user記入用ではなくツール入力用の labeling_sheet.csv を書き出す。"""
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_HEADER))
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h, "") for h in CSV_HEADER})


def _row_from_candidate(
    c: EffectFrameCandidateV3, full_path: "Path | None", crop_path: "Path | None",
) -> dict:
    """CSV/md 出力用の1候補分の辞書を組み立てる (画像パスはWindows形式)。"""
    return {
        "video_id": c.video_id, "t_sec": f"{c.t_sec:.2f}", "side": c.side,
        "layer": c.layer, "note": c.note,
        "image_full_frame": to_windows_path(full_path) if full_path else "(取得失敗)",
        "image_board_crop": to_windows_path(crop_path) if crop_path else "(取得失敗)",
    }


def _format_row_line(r: dict) -> str:
    """label_sheet.md の1候補分の行を組み立てる。"""
    layer_ja = LAYER_LABEL_JA.get(r["layer"], r["layer"])
    note = f" ({r['note']})" if r["note"] else ""
    return (
        f"- **{r['video_id']} {r['side']} t={r['t_sec']}秒** [{layer_ja}]{note} — "
        f"実画面: {r['image_full_frame']} / 盤面クロップ: {r['image_board_crop']}"
    )


def write_label_sheet_md(rows: list[dict], out_path: Path) -> Path:
    """説明書き付きの label_sheet.md を書き出す。"""
    header = [
        "# エフェクト有無 セルラベルシート 第3弾 (2026-08-04)",
        "",
        "較正 (data/verify/effect_detector_calibration_2026-08-04/calibration_report.md) "
        "で v_mean窓レベル判定が有望 (フレームAUC0.978) と判明したが n=36 は検出力不足、"
        "かつ「連鎖数テロップの発光」との混同が新失敗モードとして見つかった。実イベント"
        "から取った窓 + テロップ負例15枚を含む100フレーム規模で再較正する。",
        "",
        "## お願い",
        "- ラベル付け自体は data/verify/effect_cell_label_v3_2026-08-04/label_tool_v3.html "
        "をブラウザで開いて行ってください (このmdは一覧参考用です)。",
        "- 各フレームの盤面クロップ上で、**予告おじゃまバースト (発光) または"
        "お邪魔落下の煙が実際に被っているセル**をクリックしてマークしてください。",
        "- エフェクトが全く見えないフレームは「エフェクトなし」ボタン。",
        "- **連鎖数テロップ (「Xれんさ!」) や全消しテロップなど対象外の演出が写っている"
        "場合は「対象外エフェクト」ボタン**を押してください (これまで「スキップ」で"
        "運用していましたが、今回から専用ボタンで区別して記録します)。",
        "- フレーム自体が異常 (盤面が写っていない等) な場合のみ「フレーム異常(スキップ)」。",
        f"- 総候補数: {len(rows)} 件",
        "",
        "## 候補一覧",
        "",
    ]
    body = [_format_row_line(r) for r in rows]
    out_path.write_text("\n".join(header + body), encoding="utf-8")
    return out_path


# =============================================================================
# 7. 集計レポート
# =============================================================================


def summarize_selection(selected: list[EffectFrameCandidateV3]) -> str:
    """選定結果の分布 (レイヤー別/動画別) を平易な日本語でまとめる。"""
    by_layer: dict[str, int] = {}
    by_video: dict[str, int] = {}
    for c in selected:
        by_layer[c.layer] = by_layer.get(c.layer, 0) + 1
        by_video[c.video_stem] = by_video.get(c.video_stem, 0) + 1
    lines = [
        f"選定候補数: {len(selected)} 件 (動画数: {len(by_video)} 本)",
        f"レイヤー別: {dict(sorted(by_layer.items()))}",
        f"動画別件数: {dict(sorted(by_video.items()))}",
    ]
    return "\n".join(lines)


# =============================================================================
# メイン
# =============================================================================


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を定義・解析する。"""
    parser = argparse.ArgumentParser(description="エフェクト有無セルラベルシート第3弾準備")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    """メイン処理: 実イベント窓収集 -> 動画分散選定 -> 画像生成 -> CSV/md出力。"""
    cv2.setNumThreads(1)  # 熱対策・並列しない (アーキ指定)
    args = _parse_args()
    print("[1/3] 実イベント窓収集 + 動画分散選定 (5層: burst/smoke/telop_negative/"
          "zenkeshi/baseline)")
    selected = collect_and_select_candidates(rng_seed=args.seed)
    print("  " + summarize_selection(selected).replace("\n", "\n  "))

    print("[2/3] 画像生成")
    frames_dir = args.out_dir / FRAMES_SUBDIR_NAME
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for c in selected:
        full_path, crop_path = save_candidate_images(c, frames_dir)
        if full_path is None:
            print(f"  [WARN] {c.video_id} t={c.t_sec:.2f} {c.side}: 取得失敗、スキップ")
            continue
        rows.append(_row_from_candidate(c, full_path, crop_path))
    print(f"  画像生成完了: {len(rows)} 件")

    print("[3/3] CSV/md 出力")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_labeling_csv(rows, args.out_dir / "labeling_sheet.csv")
    sheet_path = write_label_sheet_md(rows, args.out_dir / "label_sheet.md")
    print(f"  出力: {sheet_path}")
    print(f"\n[DONE] {args.out_dir}")


if __name__ == "__main__":
    main()
