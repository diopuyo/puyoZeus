"""案B効果測定(c) 効果ゼロの構造原因診断 (2026-08-04、使い捨て診断スクリプト)。

中間結果「31/55盤面で誤り70→70 (改善ゼロ・悪化ゼロ・全盤面不変)」を受け、
ゲート自体は作動している (`scripts/_probe_c_identity2_2026-08-04.py` で
共通frameの0.39%を実際に変更したことを確認済み) にも関わらず、なぜ row1-3
(EFFECT_GATE_TOP_ROWS、誤り93セルの75.3%を占める) の誤りセルが一つも
直らなかったのかを、誤りセル単位で「焼き付いた瞬間」まで遡って構造分類する。

## 突合方式 (コピペ禁止指示に従い import 再利用)
ラベル盤面の npz 行確定は `scripts/measure_effect_gate_c_2026-08-04.py` の
アンカー突合関数 (`_lookup_anchor_row` / `_find_by_frame_idx_exact` /
`_find_bit_exact_match`) を importlib 経由で import して再利用する
(ファイル名にハイフンを含むため `from ... import` の通常構文は使えず、
`importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")` で
動的 import する)。

## 時間窓の再構成 (npz からの近似、限界を明記)
本番 `enable_effect_gate` の判定に使う「相手連鎖中」フラグ
(`chain_ev_2p is not None` 等) はフレーム単位の内部 state であり、npz には
保存されていない。npz に残るのは STABLE 確定 snapshot 行の
`chain_trigger_sec` (機能D 掛け算式検知時刻) のみなので、これを使い
以下の近似窓で「相手連鎖中」「自連鎖中」を再構成する:

    window = [trigger_sec - PRE_MARGIN_SEC, trigger_sec + POST_WINDOW_SEC]

npz には実連鎖数 (chain_count) が保存されておらず「0.4秒×連鎖数」を
厳密計算できないため、POST_WINDOW_SEC は固定近似値 (典型連鎖数3〜5+
マージン相当) を採用する。**この近似は本番判定と bit-identical ではない**
(npz の `chain_trigger_sec` は同一 trigger_sec を持つ STABLE 行が複数
連続することがあり、これは「hold 中に自分の設置で盤面が変わった」ことを
示すだけで「ゲートの時間窓がその間ずっと開いていた」ことを意味しない
ため、実際の run 中の hold 挙動を厳密には再現しない、本診断の限界として
明記する)。また **お邪魔着弾直後 window (`time_sec < _ojama_until`) は
npz から再構成不能なため本診断では検証しない** (OJAMA_FALL→STABLE
遷移の記録が npz に残らないため)。この結果 `no_time_window` は実際より
過大集計されている可能性がある (限界として stdout レポートに明記する)。

## 視覚グロー再検証
onset_t_sec ±1秒の数フレームを実動画 (`/home/ryouj/frames/video_<stem>.mp4`)
から抽出し、`src/effect_glow_detector.py` の `compute_cell_bright_ratio` /
`is_effect_glow_active` をそのまま再利用して bright_ratio_max を再計算する
(HSV 変換・閾値ロジックは一切再実装しない)。

## 分類 (優先順位、上から順に判定)
1. `pre_existing`: そのゲーム最初の STABLE 行から既に誤値 (=遷移ではない)
2. `no_time_window`: onset 時点で相手連鎖 window が開いていない
3. `own_chain_suppressed`: onset 時点で自連鎖 window が開いている
4. `glow_missed`: 時間窓は開いていたが bright_ratio_max <= 閾値 (視覚検出の見逃し)
5. `gate_should_have_fired`: 4条件すべて成立していたのに焼き付いた (実装バグ疑い)

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._diag_c_zero_effect_2026-08-04
"""
from __future__ import annotations

import csv
import importlib
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, COLOR_UNKNOWN  # noqa: E402
from src.board_state_machine import EFFECT_GATE_TOP_ROWS  # noqa: E402
from src.effect_glow_detector import (  # noqa: E402
    EFFECT_BRIGHT_RATIO_MAX_THRESHOLD,
    compute_cell_bright_ratio,
)
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402

# scripts/measure_effect_gate_c_2026-08-04.py はファイル名にハイフンを含み
# 通常の `from ... import` 構文では構文解析エラーになるため importlib で
# 動的 import する (コピペ禁止指示、突合関数を再利用するため)。
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

VIDEO_DIR: Path = Path("/home/ryouj/frames")
OUTPUT_CSV_PATH: Path = Path("data/verify/effect_gate_2026-08-04_c/diag_zero_effect.csv")

# 時間窓再構成 (npz 近似、docstring 参照)。
CHAIN_TRIGGER_DEDUP_TOL_SEC: float = 0.05  # 同一 trigger_sec の重複除去許容誤差
OPPONENT_WINDOW_PRE_MARGIN_SEC: float = 0.5
OPPONENT_WINDOW_POST_SEC: float = 2.5  # 0.4s×典型連鎖数(3〜5)+マージン相当の固定近似
OWN_WINDOW_PRE_MARGIN_SEC: float = 0.5
OWN_WINDOW_POST_SEC: float = 2.5

# 視覚グロー再検証 (onset ±1秒を数フレームだけ抽出、全フレーム走査禁止)。
GLOW_PROBE_OFFSETS_SEC: tuple[float, ...] = (
    -1.0, -0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6, 1.0,
)
FRAME_TARGET_WIDTH: int = 1920
FRAME_TARGET_HEIGHT: int = 1080

# 分類ラベル。
CAT_PRE_EXISTING: str = "pre_existing"
CAT_NO_TIME_WINDOW: str = "no_time_window"
CAT_OWN_CHAIN_SUPPRESSED: str = "own_chain_suppressed"
CAT_GLOW_MISSED: str = "glow_missed"
CAT_GATE_SHOULD_HAVE_FIRED: str = "gate_should_have_fired"
ALL_CATEGORIES: tuple[str, ...] = (
    CAT_PRE_EXISTING, CAT_NO_TIME_WINDOW, CAT_OWN_CHAIN_SUPPRESSED,
    CAT_GLOW_MISSED, CAT_GATE_SHOULD_HAVE_FIRED,
)


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class CellDiagRecord:
    """1 誤りセル分の診断結果。"""

    video: str
    side: str
    game_idx: int
    row: int
    col: int
    wrong_value: int
    correct_value: int
    label_t_sec: float
    onset_frame_idx: "int | None"
    onset_t_sec: "float | None"
    onset_count_before_label: int
    is_pre_existing: bool
    opponent_window_hit: "bool | None"
    own_chain_hit: "bool | None"
    bright_ratio_max: "float | None"
    glow_active: "bool | None"
    category: str


# =============================================================================
# 1. npz 内部での行検索・タイムライン構築
# =============================================================================


def _side_game_row_indices(idx: "object", side: str, game_idx: int) -> "np.ndarray":
    """(side, game_idx) に一致する行を frame_idx 昇順で返す。"""
    mask = (idx.sides == side) & (idx.game_idxs == game_idx)
    cand = np.where(mask)[0]
    order = np.argsort(idx.frame_idxs[cand])
    return cand[order]


def _resolve_row_index(
    idx: "object", side: str, t_sec: float, grid: "np.ndarray",
) -> "int | None":
    """突合関数が返した (grid, t_sec) から元の行 index を逆引きする。

    _find_by_frame_idx_exact / _find_bit_exact_match の判定ロジック自体は
    再実装せず、それらが確定した (grid, t_sec) を idx 配列に逆引きするだけ
    (コピペ禁止指示への対応)。
    """
    mask = (idx.sides == side) & (idx.t_secs == t_sec)
    for i in np.where(mask)[0]:
        if np.array_equal(idx.grids[i], grid):
            return int(i)
    return None


def _find_value_timeline(
    idx: "object", side: str, game_idx: int, row: int, col: int,
) -> list[tuple[int, float, int]]:
    """(frame_idx, t_sec, セル値) の時系列 (frame_idx 昇順)。"""
    rows = _side_game_row_indices(idx, side, game_idx)
    return [
        (int(idx.frame_idxs[i]), float(idx.t_secs[i]), int(idx.grids[i][row, col]))
        for i in rows
    ]


def _find_responsible_onset(
    timeline: list[tuple[int, float, int]], wrong_value: int, label_frame_idx: int,
) -> "tuple[int, float, bool, int] | None":
    """label 行の誤値に責任を持つ「直近の onset」を確定する。

    onset = 「直前の行と値が異なる (または先頭行) のに wrong_value になった」
    行。label_frame_idx 以前の onset のうち最後のものを採用する。
    途中で治って再発した場合は onset_count_before_label > 1 になる。
    """
    onsets: list[tuple[int, int, float]] = []  # (timeline_idx, frame_idx, t_sec)
    prev_val: "int | None" = None
    for i, (fi, t, v) in enumerate(timeline):
        if v == wrong_value and (i == 0 or prev_val != wrong_value):
            onsets.append((i, fi, t))
        prev_val = v
    valid = [o for o in onsets if o[1] <= label_frame_idx]
    if not valid:
        return None
    last = valid[-1]
    is_pre_existing = last[0] == 0
    return last[1], last[2], is_pre_existing, len(valid)


# =============================================================================
# 2. 時間窓再構成 (chain_trigger_sec からの近似)
# =============================================================================


def _load_chain_trigger_secs(npz_path: Path) -> "np.ndarray":
    """npz から chain_trigger_sec 列を生配列で読む (_NpzIndex は本列を持たないため別読込)。

    _load_npz_index は行の並び替えを行わないため、ここで読む配列は
    _NpzIndex の各配列と行 index が一致する (整合性はそのまま利用できる)。
    """
    data = np.load(npz_path, allow_pickle=True)
    return data["chain_trigger_sec"].astype(np.float64)


def _build_trigger_windows(
    idx: "object", chain_trigger_secs: "np.ndarray", side: str, game_idx: int,
    pre_margin_sec: float, post_window_sec: float,
) -> list[tuple[float, float]]:
    """chain_trigger_sec の異なる値ごとに近似時間窓を作る (docstring の限界を参照)。"""
    rows = _side_game_row_indices(idx, side, game_idx)
    windows: list[tuple[float, float]] = []
    seen: list[float] = []
    for i in rows:
        v = float(chain_trigger_secs[i])
        if math.isnan(v):
            continue
        if any(math.isclose(v, sv, abs_tol=CHAIN_TRIGGER_DEDUP_TOL_SEC) for sv in seen):
            continue
        seen.append(v)
        windows.append((v - pre_margin_sec, v + post_window_sec))
    return windows


def _time_in_any_window(t_sec: float, windows: list[tuple[float, float]]) -> bool:
    """t_sec がいずれかの (start, end) 窓に含まれるか。"""
    return any(start <= t_sec <= end for start, end in windows)


# =============================================================================
# 3. 視覚グロー再検証 (動画から数フレーム抽出)
# =============================================================================


def _max_bright_ratio_in_rows(
    frame_bgr: "np.ndarray", region: "object", rows: "frozenset[int]",
) -> float:
    """rows × BOARD_COLS の bright_ratio 最大値 (effect_glow_detector の集約方式と同一)。"""
    max_ratio = 0.0
    for row in rows:
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            ratio = compute_cell_bright_ratio(frame_bgr[y1:y2, x1:x2])
            if ratio > max_ratio:
                max_ratio = ratio
    return max_ratio


def _read_frame_at(cap: "cv2.VideoCapture", fps: float, t_sec: float) -> "np.ndarray | None":
    """指定秒にシークして 1920x1080 リサイズ済みフレームを読む。"""
    if t_sec < 0:
        return None
    frame_no = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_no))
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    if frame.shape[1] != FRAME_TARGET_WIDTH or frame.shape[0] != FRAME_TARGET_HEIGHT:
        frame = cv2.resize(frame, (FRAME_TARGET_WIDTH, FRAME_TARGET_HEIGHT))
    return frame


def _probe_glow(
    cap: "cv2.VideoCapture", fps: float, side: str, onset_t_sec: float,
) -> tuple[float, bool]:
    """onset ±1秒の数フレームから bright_ratio_max を再計算する (全フレーム走査禁止)。"""
    region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    best = 0.0
    for offset in GLOW_PROBE_OFFSETS_SEC:
        frame = _read_frame_at(cap, fps, onset_t_sec + offset)
        if frame is None:
            continue
        ratio = _max_bright_ratio_in_rows(frame, region, EFFECT_GATE_TOP_ROWS)
        if ratio > best:
            best = ratio
    return best, best > EFFECT_BRIGHT_RATIO_MAX_THRESHOLD


# =============================================================================
# 4. 分類
# =============================================================================


def classify_cell(
    is_pre_existing: bool,
    opponent_window_hit: bool,
    own_chain_hit: bool,
    glow_active: "bool | None",
) -> str:
    """優先順位に従って1誤りセルを分類する (docstring の分類定義参照)。"""
    if is_pre_existing:
        return CAT_PRE_EXISTING
    if not opponent_window_hit:
        return CAT_NO_TIME_WINDOW
    if own_chain_hit:
        return CAT_OWN_CHAIN_SUPPRESSED
    if not glow_active:
        return CAT_GLOW_MISSED
    return CAT_GATE_SHOULD_HAVE_FIRED


# =============================================================================
# 5. メイン診断ループ
# =============================================================================


def _mismatched_top_row_cells(
    c_grid: "np.ndarray", correct_grid: "np.ndarray",
) -> list[tuple[int, int, int, int]]:
    """row1-3 かつ correct!=U の誤りセルを (row, col, wrong, correct) で列挙する。"""
    out: list[tuple[int, int, int, int]] = []
    for row in EFFECT_GATE_TOP_ROWS:
        for col in range(BOARD_COLS):
            cv = int(correct_grid[row, col])
            if cv == COLOR_UNKNOWN:
                continue
            wv = int(c_grid[row, col])
            if wv != cv:
                out.append((row, col, wv, cv))
    return out


def _diagnose_one_cell(
    c_idx: "object", chain_trigger_secs: "np.ndarray", side: str, row_game_idx: int,
    label_frame_idx: int, label_t_sec: float, row: int, col: int, wrong: int,
    correct: int, cap: "cv2.VideoCapture", fps: float,
) -> CellDiagRecord:
    """1 誤りセルを onset 特定 → 時間窓/自連鎖判定 → 視覚グロー判定 → 分類する。"""
    timeline = _find_value_timeline(c_idx, side, row_game_idx, row, col)
    onset = _find_responsible_onset(timeline, wrong, label_frame_idx)
    if onset is None:
        # 想定外 (label 行自体が timeline に含まれないはずがない)。異常として記録。
        return CellDiagRecord(
            "", side, row_game_idx, row, col, wrong, correct, label_t_sec,
            None, None, 0, False, None, None, None, None, "onset_not_found",
        )
    onset_fi, onset_t, is_pre, n_onsets = onset
    if is_pre:
        return CellDiagRecord(
            "", side, row_game_idx, row, col, wrong, correct, label_t_sec,
            onset_fi, onset_t, n_onsets, True, None, None, None, None,
            CAT_PRE_EXISTING,
        )
    opp_side = "2P" if side == "1P" else "1P"
    opp_windows = _build_trigger_windows(
        c_idx, chain_trigger_secs, opp_side, row_game_idx,
        OPPONENT_WINDOW_PRE_MARGIN_SEC, OPPONENT_WINDOW_POST_SEC,
    )
    own_windows = _build_trigger_windows(
        c_idx, chain_trigger_secs, side, row_game_idx,
        OWN_WINDOW_PRE_MARGIN_SEC, OWN_WINDOW_POST_SEC,
    )
    opp_hit = _time_in_any_window(onset_t, opp_windows)
    own_hit = _time_in_any_window(onset_t, own_windows)
    bright_max: "float | None" = None
    glow_active: "bool | None" = None
    if opp_hit and not own_hit:
        bright_max, glow_active = _probe_glow(cap, fps, side, onset_t)
    category = classify_cell(False, opp_hit, own_hit, glow_active)
    return CellDiagRecord(
        "", side, row_game_idx, row, col, wrong, correct, label_t_sec,
        onset_fi, onset_t, n_onsets, False, opp_hit, own_hit, bright_max,
        glow_active, category,
    )


def _diagnose_one_video(video_stem: str) -> list[CellDiagRecord]:
    """1 動画分の fixed ラベル盤面から row1-3 誤りセルを全て診断する。"""
    c_path = _MC.NPZ_DIR_C / f"{video_stem}.npz"
    if not c_path.exists():
        return []  # 未着弾動画は明示スキップ
    anchor_idx = _MC._load_npz_index(_MC.ANCHOR_NPZ_DIR / f"{video_stem}.npz")
    c_idx = _MC._load_npz_index(c_path)
    chain_trigger_secs = _load_chain_trigger_secs(c_path)
    samples = [
        s for s in _MC.load_all_samples()
        if s.video_stem == video_stem and s.status == "fixed"
    ]
    video_path = VIDEO_DIR / f"video_{video_stem}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    records: list[CellDiagRecord] = []
    for s in samples:
        records.extend(
            _diagnose_one_sample(s, anchor_idx, c_idx, chain_trigger_secs, cap, fps)
        )
    cap.release()
    for r in records:
        r.video = video_stem
    return records


def _diagnose_one_sample(
    s: "object", anchor_idx: "object", c_idx: "object", chain_trigger_secs: "np.ndarray",
    cap: "cv2.VideoCapture", fps: float,
) -> list[CellDiagRecord]:
    """1 ラベルサンプル分の row1-3 誤りセルを診断する (突合失敗は空リストで明示スキップ)。"""
    anchor = _MC._lookup_anchor_row(
        anchor_idx, s.side, s.t_sec, s.game_idx, s.anchor_recognized_grid,
    )
    if anchor is None:
        print(f"  [警告] アンカー突合失敗: {s.video_stem} t={s.t_sec} {s.side}")
        return []
    match = _MC._find_by_frame_idx_exact(c_idx, s.side, anchor.frame_idx)
    if match is None:
        match = _MC._find_bit_exact_match(
            c_idx, s.side, anchor.grid, anchor.t_sec, _MC.ANCHOR_MATCH_WINDOW_SEC,
        )
    if match is None:
        print(f"  [警告] (c) no_match: {s.video_stem} t={s.t_sec} {s.side}")
        return []
    c_grid, c_t = match
    c_row_index = _resolve_row_index(c_idx, s.side, c_t, c_grid)
    if c_row_index is None:
        print(f"  [警告] (c) 行逆引き失敗: {s.video_stem} t={s.t_sec} {s.side}")
        return []
    row_game_idx = int(c_idx.game_idxs[c_row_index])
    label_frame_idx = int(c_idx.frame_idxs[c_row_index])
    cells = _mismatched_top_row_cells(c_grid, s.correct_grid)
    return [
        _diagnose_one_cell(
            c_idx, chain_trigger_secs, s.side, row_game_idx, label_frame_idx, c_t,
            row, col, wrong, correct, cap, fps,
        )
        for row, col, wrong, correct in cells
    ]


# =============================================================================
# 6. 集計レポート + CSV 出力
# =============================================================================


def write_diag_csv(records: list[CellDiagRecord], out_path: Path) -> None:
    """セル単位の明細 CSV を出力する。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video", "side", "game_idx", "row", "col", "wrong_value", "correct_value",
        "label_t_sec", "onset_frame_idx", "onset_t_sec", "onset_count_before_label",
        "is_pre_existing", "opponent_window_hit", "own_chain_hit",
        "bright_ratio_max", "glow_active", "category",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: getattr(r, k) for k in fieldnames})


def build_category_summary(records: list[CellDiagRecord]) -> str:
    """分類別集計 + bright_ratio_max 分布 + 「エフェクト起因でない誤り」割合。"""
    lines = ["--- 分類別集計 ---"]
    n = len(records)
    for cat in ALL_CATEGORIES:
        c = sum(1 for r in records if r.category == cat)
        pct = (c / n * 100) if n else 0.0
        lines.append(f"  {cat:22} {c:>4} 件 ({pct:5.1f}%)")
    n_other = sum(1 for r in records if r.category not in ALL_CATEGORIES)
    if n_other:
        lines.append(f"  {'(その他/異常)':22} {n_other:>4} 件")
    n_non_effect = sum(
        1 for r in records if r.category in (CAT_PRE_EXISTING, CAT_NO_TIME_WINDOW)
    )
    pct_non_effect = (n_non_effect / n * 100) if n else 0.0
    lines.append("")
    lines.append(
        f"[重要] エフェクト起因でない誤り (pre_existing + no_time_window): "
        f"{n_non_effect}/{n} 件 ({pct_non_effect:.1f}%)"
    )
    n_bug = sum(1 for r in records if r.category == CAT_GATE_SHOULD_HAVE_FIRED)
    lines.append(f"[重要] 実装バグ疑い (gate_should_have_fired): {n_bug} 件")
    ratios = [r.bright_ratio_max for r in records if r.bright_ratio_max is not None]
    if ratios:
        lines.append("")
        lines.append(
            f"bright_ratio_max 分布 (視覚判定に到達した {len(ratios)} 件): "
            f"min={min(ratios):.3f} median={statistics.median(ratios):.3f} "
            f"max={max(ratios):.3f} (閾値={EFFECT_BRIGHT_RATIO_MAX_THRESHOLD})"
        )
    return "\n".join(lines)


def main() -> None:
    landed = sorted(p.stem for p in _MC.NPZ_DIR_C.glob("*.npz"))
    print(f"[1/3] (c) 着弾済み動画: {len(landed)} 本 ({landed})")

    records: list[CellDiagRecord] = []
    for stem in landed:
        recs = _diagnose_one_video(stem)
        records.extend(recs)
        print(f"  {stem}: row1-3 誤りセル {len(recs)} 件")

    print(f"\n[2/3] 診断対象セル総数: {len(records)} 件")
    write_diag_csv(records, OUTPUT_CSV_PATH)
    print(f"[出力] セル単位明細 CSV: {OUTPUT_CSV_PATH}")

    print("\n[3/3] 集計")
    print(build_category_summary(records))
    print(
        "\n[限界] お邪魔着弾直後 window は npz から再構成不能なため未検証 "
        "(no_time_window は過大集計の可能性あり)。時間窓は chain_trigger_sec "
        "からの固定近似 (0.4秒×連鎖数の厳密計算は不可、docstring 参照)。"
    )


if __name__ == "__main__":
    main()
