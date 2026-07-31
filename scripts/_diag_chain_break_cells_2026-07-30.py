"""連鎖を切っている「犯人セル」特定+類型化 (2026-07-30)。

## 背景

simulate() (src/chain.py の ChainSimulator) は連鎖数を一方向に過小評価し、
真の連鎖数が大きいほど誤差が拡大する事例が実フレーム目視で確認された
(F2/E/I の3件、memory project_chain_count_both_untrustworthy_2026-07-30)。
原因は simulate() 自体のバグでなく、入力の before_board (認識盤面) が
本来つながっている連結を繋げられていないこと (認識の欠損) と推定されている。
本スクリプトは推測でなく、「連鎖が止まった盤面」上の near-miss (あと1個で
発火する3連結) を機械的に列挙し、その隣接セルを犯人候補として類型化する
(推測結論は書かない。実フレーム確認は代表事例のみ別途行う)。

## アプローチ選定 (near-miss、ChainSimulator採用の理由)

`src/chain_bitboard.py` の `simulate_batch` はバッチ高速化のため色ごとに
「4連結以上の union mask」のみを返し、個々のグループの厳密なメンバーシップ・
サイズ (特に size==3 の劣角グループ) を保持しない。近道分析には
「あと1個で発火する3連結」を個別グループとして特定する必要があるため、
本スクリプトは `src.chain.ChainSimulator` (BFS flood fill、`find_groups()` が
size 問わず全グループを厳密に返す) を採用する。simulate()/find_groups() は
docstring に stateless (引数 board を破壊しない) と明記されており、
既存 API を一切変更せず読み取り専用で呼び出すだけで要件を満たせる。

## 使用データ世代 (厳守事項)

- before_board 復元: `data/indicators_v2/boards_lean_fixed_regen_2026-07-28`
  (#51修正後、7/28生成。着弾CSV `exchange_landing_delay_regen_2026-07-28.csv` と
  同じ世代)。`scripts/measure_exchange_dynamics.py` の NPZ_DIR 既定
  (`boards_lean_fixed`、7/18の旧npz) とは別物であり、本スクリプトでは使わない。
- underestimate 候補抽出元: `data/verify/chain_count_ocr_full_corpus_2026-07-29.csv`
  (走行中・追記されるため読み取り専用でオープンし、ジョブを止めない)。

## 実行方法

WSL 経由、nice -n 19、単一プロセス (並列で2系統走っているためCPUを食わない):
    nice -n 19 PYTHONPATH=. ./venv/bin/python \
        scripts/_diag_chain_break_cells_2026-07-30.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS,
    COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN, Board,
)
from src.chain import ChainSimulator, NEIGHBOR_DELTAS, PuyoGroup  # noqa: E402
from src.chain_count_ocr import _ensure_1080p  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION, BoardRegion  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    FireEvent, NpzRecord, _load_npz, _process_video, _subset,
)

# ============================
# 定数
# ============================

NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
CORPUS_CSV: Path = PROJ_ROOT / "data" / "verify" / "chain_count_ocr_full_corpus_2026-07-29.csv"
OUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "chain_break_cells_2026-07-30"

# 「あと1個で発火する」近道グループのサイズ (MIN_ERASE_COUNT(4) - 1)。
NEAR_MISS_GROUP_SIZE: int = 3

# CSV由来の機械的near-miss対象抽出の閾値 (screen_chain_count - new_chain_count)。
# 2以上を「simulate過小評価の疑いが強い」候補とする (n=90、2026-07-30実測)。
UNDERESTIMATE_GAP_MIN: int = 2

# セルの「最近変化した」判定の閾値秒 (これ以下なら直近変化、それ以上は既存構造)。
# 根拠なき断定を避けるため、値そのものも常に併記して報告する。
FRESHNESS_RECENT_THRESHOLD_SEC: float = 2.0

# 画像出力の拡大倍率 (盤面全体 / 単セル)
BOARD_CROP_UPSCALE: int = 2
CELL_CROP_UPSCALE: int = 10


# ============================
# データクラス
# ============================


@dataclass(frozen=True)
class TargetEvent:
    """near-miss 分析の対象イベント (代表事例 or CSV由来の機械抽出)。"""
    video_stem: str
    side: str
    game_idx: int
    t_chain_start_key: float  # round(t_chain_start, 1) でイベント辞書と突き合わせる
    label: str
    new_chain_count_csv: float
    screen_chain_count_csv: float
    is_representative: bool


@dataclass(frozen=True)
class NearMissCandidate:
    """あと1個で発火する3連結の隣接「犯人候補」セル。"""
    group_color: int
    group_cells: tuple[tuple[int, int], ...]
    neighbor_row: int
    neighbor_col: int
    neighbor_value: int


# 2026-07-30 タスク指定の代表4事例 (F2/E/I/G)。Gはsimulateが正解した負の対照。
REPRESENTATIVE_TARGETS: tuple[TargetEvent, ...] = (
    TargetEvent("c21", "2P", 6, 584.2, "F2_c21_2P_g6", 1.0, 8.0, True),
    TargetEvent("c21", "1P", 2, 292.6, "E_c21_1P_g2", 2.0, 8.0, True),
    TargetEvent("c22", "1P", 1, 286.0, "I_c22_1P_g1", 4.0, 9.0, True),
    TargetEvent("c22", "2P", 6, 641.0, "G_c22_2P_g6_negctrl", 13.0, 9.0, True),
)


# ============================
# npz / FireEvent ヘルパ (既存スクリプトの手順を複製、読み取り専用)
# ============================
#
# scripts/_verify_chain_count_ocr_full_corpus_2026-07-29.py の
# _events_for_stem() / _reconstruct_before_board() と同一手順。
# その関数はファイル名にハイフンを含み import 不可能なため複製する
# (既存ファイルは変更しない、CLAUDE.md backward compat 原則)。


def _events_for_stem(stem: str) -> dict[tuple[str, int, float], FireEvent]:
    """stem の de-frag 後 FireEvent を (side, game_idx, t_chain_start概算) キーで返す。"""
    sim = ChainSimulator()
    npz_path = NPZ_DIR_REGEN / f"{stem}.npz"
    if not npz_path.exists():
        return {}
    _, defrag, _ = _process_video(npz_path, sim, 0)
    return {(e.fire_side, e.game_idx, round(e.t_chain_start, 1)): e for e in defrag}


def _load_records_by_side(stem: str) -> dict[str, NpzRecord]:
    """stem の npz を側ごとの NpzRecord 辞書として返す (空なら該当なし)。"""
    npz_path = NPZ_DIR_REGEN / f"{stem}.npz"
    if not npz_path.exists():
        return {}
    return {rec.side: rec for rec in _load_npz(npz_path)}


def _game_subset(records: dict[str, NpzRecord], ev: FireEvent) -> NpzRecord | None:
    """イベントの (side, game_idx) に絞った NpzRecord を返す。"""
    if ev.fire_side not in records:
        return None
    rec = records[ev.fire_side]
    mask = rec.game_idx == ev.game_idx
    return _subset(rec, mask)


def _reconstruct_before_board(g: NpzRecord, ev: FireEvent) -> Board | None:
    """de-frag 後 FireEvent の before_idx から before_board を復元する。"""
    if ev.before_idx < 0 or ev.before_idx >= len(g.t_sec):
        return None
    return Board.from_list(g.grids[ev.before_idx].tolist())


def _seconds_since_last_change(
    g: NpzRecord, before_idx: int, cell: tuple[int, int],
) -> float | None:
    """指定セルが現在値になってから何秒経過したか (履歴window内で不変なら None)。

    None は「このゲームの認識window開始時点から既にこの値だった (それ以前は
    不明)」を意味する (=既存構造である可能性が高いことの弱い代理指標。
    強い断定はしない、報告時に必ず値をそのまま併記する)。
    """
    row, col = cell
    current = int(g.grids[before_idx, row, col])
    idx = before_idx
    while idx > 0 and int(g.grids[idx - 1, row, col]) == current:
        idx -= 1
    if idx == 0:
        return None
    return float(g.t_sec[before_idx] - g.t_sec[idx])


# ============================
# near-miss 検出・類型化
# ============================


def _find_near_miss_candidates(
    board: Board, sim: ChainSimulator,
) -> list[NearMissCandidate]:
    """盤面上の size==3 グループに隣接する「犯人候補」セルを列挙する。

    同色の隣接セルは find_groups() の flood fill 時点で既にグループに
    吸収されているはずなので、ここで列挙される隣接セルは理論上すべて
    「グループ色と異なる値」を持つ (空/UNKNOWN/別色/おじゃま)。
    """
    candidates: list[NearMissCandidate] = []
    for group in sim.find_groups(board):
        if group.size != NEAR_MISS_GROUP_SIZE:
            continue
        seen: set[tuple[int, int]] = set()
        for (row, col) in group.cells:
            for dr, dc in NEIGHBOR_DELTAS:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < BOARD_ROWS and 0 <= nc < BOARD_COLS):
                    continue
                if (nr, nc) in group.cells or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                candidates.append(NearMissCandidate(
                    group_color=group.color,
                    group_cells=tuple(sorted(group.cells)),
                    neighbor_row=nr, neighbor_col=nc,
                    neighbor_value=board.get(nr, nc),
                ))
    return candidates


def _classify_category(value: int) -> str:
    """犯人候補セルの値を誤りの種類に分類する。"""
    if value == COLOR_EMPTY:
        return "色→空(未反映)"
    if value == COLOR_UNKNOWN:
        return "色→UNKNOWN"
    if value == COLOR_OJAMA:
        return "おじゃま絡み"
    return "色→別色(誤読)"


def _row_band(row: int) -> str:
    """盤面上の位置 (行) を4帯域に分類する。"""
    if row < HIDDEN_ROWS:
        return "隠し段(row0)"
    if row <= 3:
        return "上部(row1-3)"
    if row <= 8:
        return "中央(row4-8)"
    return "下部(row9-12)"


def _col_band(col: int) -> str:
    """盤面上の位置 (列) を端/中央に分類する。"""
    return "端(col0/5)" if col in (0, BOARD_COLS - 1) else "中央(col1-4)"


def _freshness_bucket(delta_sec: float | None) -> str:
    """freshness (直近変化からの経過秒) を粗いバケットに分類する。"""
    if delta_sec is None:
        return "window開始時点から不変(既存構造の疑い、断定不可)"
    if delta_sec <= FRESHNESS_RECENT_THRESHOLD_SEC:
        return f"直近変化(<={FRESHNESS_RECENT_THRESHOLD_SEC}秒)"
    return f"既存構造寄り(>{FRESHNESS_RECENT_THRESHOLD_SEC}秒)"


# ============================
# CSV 由来の機械抽出対象 (simulate 過小評価の疑いが強い事例)
# ============================


def _build_targets_from_corpus() -> list[TargetEvent]:
    """chain_count_ocr_full_corpus から gap>=UNDERESTIMATE_GAP_MIN の事例を抽出する。

    読み取り専用でオープンする (走行中に追記されるファイル、書き込み禁止)。
    """
    df = pd.read_csv(CORPUS_CSV)
    sub = df[
        df["new_chain_count"].notna() & df["screen_chain_count"].notna()
        & (df["video_available"] == True)  # noqa: E712
    ].copy()
    sub["gap"] = sub["screen_chain_count"] - sub["new_chain_count"]
    sub = sub[sub["gap"] >= UNDERESTIMATE_GAP_MIN]
    targets: list[TargetEvent] = []
    for _, row in sub.iterrows():
        label = (
            f"{row['video_stem']}_{row['side']}_g{int(row['game_idx'])}"
            f"_t{row['t_chain_start']:.1f}"
        )
        targets.append(TargetEvent(
            video_stem=str(row["video_stem"]), side=str(row["side"]),
            game_idx=int(row["game_idx"]),
            t_chain_start_key=round(float(row["t_chain_start"]), 1),
            label=label,
            new_chain_count_csv=float(row["new_chain_count"]),
            screen_chain_count_csv=float(row["screen_chain_count"]),
            is_representative=False,
        ))
    return targets


def _merge_targets(csv_targets: list[TargetEvent]) -> list[TargetEvent]:
    """代表4事例 + CSV由来事例を重複排除して統合する (代表事例を優先)。"""
    dedup: dict[tuple[str, str, int, float], TargetEvent] = {}
    for t in list(REPRESENTATIVE_TARGETS) + csv_targets:
        key = (t.video_stem, t.side, t.game_idx, t.t_chain_start_key)
        if key not in dedup or t.is_representative:
            dedup[key] = t
    return list(dedup.values())


# ============================
# 実フレーム画像出力 (代表事例のみ、userレビュー用に判読可能な解像度で)
# ============================


def _crop_board_region(frame: np.ndarray, side: str) -> np.ndarray:
    """盤面全体 (可視12行) を切り出す。"""
    region: BoardRegion = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    return frame[region.y:region.y + region.height,
                 region.x:region.x + region.width].copy()


def _crop_cell_region(frame: np.ndarray, side: str, row: int, col: int) -> np.ndarray:
    """指定セル1個分 (隠し段含め cell_center 基準) を切り出す。"""
    region: BoardRegion = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
    cx, cy = region.cell_center(row, col)
    half_w = max(1, int(region.cell_width / 2))
    half_h = max(1, int(region.cell_height / 2))
    y1, y2 = max(0, cy - half_h), cy + half_h
    x1, x2 = max(0, cx - half_w), cx + half_w
    return frame[y1:y2, x1:x2].copy()


def _upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """判読性のため最近傍拡大する (色境界をぼかさない)。"""
    if img.size == 0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def _grab_frame(video_path: Path, t_sec: float) -> np.ndarray | None:
    """指定時刻のフレームを取得し 1080p に正規化する (c11/c21 等 720p 動画対策)。"""
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return _ensure_1080p(frame)


def _save_representative_images(
    tgt: TargetEvent, ev: FireEvent, candidates: list[NearMissCandidate],
) -> None:
    """代表事例の盤面全体+犯人候補セルの拡大画像を保存する (userレビュー用)。"""
    video_path = VIDEO_DIR / f"video_{tgt.video_stem}.mp4"
    frame = _grab_frame(video_path, ev.t_chain_start)
    if frame is None:
        print(f"[WARN] frame取得失敗: {tgt.label} (t={ev.t_chain_start})")
        return
    ev_dir = OUT_DIR / tgt.label
    ev_dir.mkdir(parents=True, exist_ok=True)
    board_crop = _crop_board_region(frame, tgt.side)
    cv2.imwrite(str(ev_dir / "board_full.png"), _upscale(board_crop, BOARD_CROP_UPSCALE))
    seen: set[tuple[int, int]] = set()
    for c in candidates:
        cell = (c.neighbor_row, c.neighbor_col)
        if cell in seen:
            continue
        seen.add(cell)
        crop = _crop_cell_region(frame, tgt.side, c.neighbor_row, c.neighbor_col)
        fname = f"cell_r{c.neighbor_row}_c{c.neighbor_col}_val{c.neighbor_value}.png"
        cv2.imwrite(str(ev_dir / fname), _upscale(crop, CELL_CROP_UPSCALE))
    print(f"[OK] {tgt.label}: 画像保存 {ev_dir} (候補セル{len(seen)}個)")


# ============================
# per-target / per-video 処理
# ============================


def _row_dict(
    tgt: TargetEvent, ev: FireEvent, recomputed_chain_count: int,
    c: NearMissCandidate, freshness_sec: float | None,
) -> dict:
    """1犯人候補セル分の出力行を組み立てる。"""
    return {
        "video_stem": tgt.video_stem, "side": tgt.side, "game_idx": tgt.game_idx,
        "t_chain_start": ev.t_chain_start, "label": tgt.label,
        "is_representative": tgt.is_representative,
        "recomputed_chain_count": recomputed_chain_count,
        "csv_new_chain_count": tgt.new_chain_count_csv,
        "csv_screen_chain_count": tgt.screen_chain_count_csv,
        "group_color": c.group_color,
        "group_cells": str(c.group_cells),
        "neighbor_row": c.neighbor_row, "neighbor_col": c.neighbor_col,
        "neighbor_value": c.neighbor_value,
        "category": _classify_category(c.neighbor_value),
        "row_band": _row_band(c.neighbor_row),
        "col_band": _col_band(c.neighbor_col),
        "freshness_sec": freshness_sec,
        "freshness_bucket": _freshness_bucket(freshness_sec),
    }


def _resolve_event(
    tgt: TargetEvent, events: dict[tuple[str, int, float], FireEvent],
    records: dict[str, NpzRecord],
) -> tuple[FireEvent, NpzRecord, Board] | None:
    """target から FireEvent・game subset・before_board を解決する。"""
    key = (tgt.side, tgt.game_idx, tgt.t_chain_start_key)
    ev = events.get(key)
    if ev is None:
        print(f"[WARN] event not found: {tgt.label} key={key}")
        return None
    g = _game_subset(records, ev)
    if g is None:
        print(f"[WARN] game subset not found: {tgt.label}")
        return None
    before_board = _reconstruct_before_board(g, ev)
    if before_board is None:
        print(f"[WARN] before_board reconstruct失敗: {tgt.label}")
        return None
    return ev, g, before_board


def _process_one_target(
    tgt: TargetEvent, events: dict[tuple[str, int, float], FireEvent],
    records: dict[str, NpzRecord], sim: ChainSimulator,
) -> list[dict]:
    """1イベント分の near-miss 候補行を生成する (代表事例は画像保存も行う)。"""
    resolved = _resolve_event(tgt, events, records)
    if resolved is None:
        return []
    ev, g, before_board = resolved
    result = sim.simulate(before_board)
    if tgt.new_chain_count_csv == tgt.new_chain_count_csv:  # NaN でない
        if result.chain_count != int(tgt.new_chain_count_csv):
            print(f"[NOTE] {tgt.label}: 再計算chain_count={result.chain_count} "
                  f"!= CSV new_chain_count={tgt.new_chain_count_csv} (npz世代差の疑い)")
    candidates = _find_near_miss_candidates(result.final_board, sim)
    if tgt.is_representative:
        _save_representative_images(tgt, ev, candidates)
    rows: list[dict] = []
    for c in candidates:
        freshness = _seconds_since_last_change(g, ev.before_idx, (c.neighbor_row, c.neighbor_col))
        rows.append(_row_dict(tgt, ev, result.chain_count, c, freshness))
    return rows


def _process_video_targets(
    stem: str, targets: list[TargetEvent], sim: ChainSimulator,
) -> list[dict]:
    """1動画分の全 target を処理する。"""
    events = _events_for_stem(stem)
    records = _load_records_by_side(stem)
    rows: list[dict] = []
    for tgt in targets:
        rows.extend(_process_one_target(tgt, events, records, sim))
    return rows


# ============================
# 層別集計 (「代表値を出す前に層別せよ」原則)
# ============================


def _print_and_save_summary(df: pd.DataFrame) -> None:
    """層別軸ごとの集計を表示・保存する (プールした代表値は出さない)。"""
    if df.empty:
        print("[WARN] 候補セルが0件。近道分析は空振り。")
        return
    axes = ["category", "row_band", "col_band", "side", "video_stem", "freshness_bucket"]
    for axis in axes:
        counts = df.groupby(axis).size().sort_values(ascending=False)
        print(f"\n[{axis} 別件数]\n{counts.to_string()}")
        counts.to_csv(OUT_DIR / f"summary_by_{axis}.csv", header=["count"])
    cross = df.groupby(["category", "side"]).size().unstack(fill_value=0)
    print(f"\n[category x side クロス集計]\n{cross.to_string()}")
    cross.to_csv(OUT_DIR / "summary_category_x_side.csv")
    n_events = df.groupby(["video_stem", "side", "game_idx", "label"]).ngroups
    print(f"\n[集計対象] イベント数={n_events}件, 犯人候補セル総数={len(df)}件")


# ============================
# main
# ============================


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_targets = _build_targets_from_corpus()
    print(f"[準備] CSV由来underestimate候補: {len(csv_targets)}件 "
          f"(gap>={UNDERESTIMATE_GAP_MIN})、代表事例: {len(REPRESENTATIVE_TARGETS)}件")
    targets_all = _merge_targets(csv_targets)

    by_stem: dict[str, list[TargetEvent]] = {}
    for t in targets_all:
        by_stem.setdefault(t.video_stem, []).append(t)

    sim = ChainSimulator()
    all_rows: list[dict] = []
    for stem, targets in sorted(by_stem.items()):
        print(f"\n[{stem}] {len(targets)}件処理中...", flush=True)
        rows = _process_video_targets(stem, targets, sim)
        all_rows.extend(rows)
        print(f"[{stem}] 完了: 候補セル{len(rows)}件", flush=True)

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "candidate_cells.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n保存: {out_csv} ({len(df)}行)")
    _print_and_save_summary(df)


if __name__ == "__main__":
    main()
