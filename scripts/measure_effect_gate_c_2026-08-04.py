"""案B効果測定(c): 4条件フルON (enable_effect_gate + enable_effect_visual_gate)
の誤りセル削減効果を batch1+batch2 統合 55 盤面ラベルで測定する (2026-08-04)。

## scripts/measure_effect_gate_impact_2026-08-03.py (v3) からの差分

1. **ラベル対象を batch1+batch2 に拡張**: v3 は batch1
   (`data/verify/full_board_label_sheet_2026-08-02/`, 27件) のみを対象にして
   いたが、本スクリプトは batch2
   (`data/verify/full_board_label_sheet_batch2_2026-08-03/`, 28件) も統合し、
   計 55 盤面 (ok/fixed のみ、skip 除外) を対象にする。各サンプルに
   `batch` タグを付与し、層別集計 (feedback_stratify_before_pooling) を
   徹底する。
2. **OFF 基準 = アンカー npz そのものを再利用**: v3 は OFF 用に別途再収集した
   npz (`data/verify/effect_gate_2026-08-03_v2/off/`) を用意していたが、
   本スクリプトはラベル元の本番 npz
   (`data/indicators_v2/boards_lean_regen_2026-07-31/`) をアンカーかつ OFF
   基準として直接使う (アンカー突合で確定した行そのものが OFF の値)。
   再収集コスト・pipeline 設定ズレのリスクを排除する。
3. **比較対象は2系統**:
   - (b) 時間ゲートのみ (`enable_effect_gate`単体):
     `data/verify/effect_gate_2026-08-03_v2/on/` (batch1 の24動画のみ存在。
     batch2 専用動画 [c16/c31/c34/c36/c37/c42/c44] は npz 不在 = 明示 N/A)
   - (c) 4条件フルAND (`enable_effect_gate`+`enable_effect_visual_gate`):
     `data/verify/effect_gate_2026-08-04_c/on_full/` (31動画対象、裏で再認識
     走行中。未着弾動画は「未着弾」として明示スキップし、着弾分のみ集計する)
4. **OFF アンカー→(b)/(c) 突合は frame_idx 完全一致検索 (v3 のアンカー→OFF
   突合と同一技法) を主方式にする**: v3 は OFF→ON の突合には「時刻窓内で
   最も近い snapshot」(`_find_nearest_snapshot`、常に何かに一致してしまう
   fail-silent 気味の方式) を使っていたが、本スクリプトはそれを使わない。
   代わりに v3 が「アンカー npz → OFF 再収集 npz」の突合で使っていた
   **frame_idx 完全一致 → 失敗時のみ bit-exact 時刻窓フォールバック**の技法を
   「OFF アンカー → (b)/(c) npz」に適用する。bit-exact フォールバックは
   「(b)/(c) npz の grid が OFF (=アンカー) と完全一致する行」を探すため、
   ゲートが実際に値を変えた (=効果があった) ケースでは原理的に一致せず、
   常に frame_idx 完全一致側で検出される (bit-exact 一致が成立するのは
   「その時刻帯で内容が変わっていない」場合のみなので、真の改善/劣化を
   隠すことはない)。フォールバックは単純に dedup 間引きタイミングの差で
   frame_idx がわずかにずれた「無変化」ケースを回収するだけである。どちらの
   段でも一致行が見つからない場合のみ「no_match」として明示報告する (N/A
   [npz不在/未着弾] とは別カテゴリで区別する、fail-silent 回避)。
5. **誤り分類 (色→9誤検出の減少を主目的とする罪の序列に対応)**:
   memory `project_full_board_error_taxonomy_2026-08-02` の罪の序列
   (色→おじゃま誤検出が最も重い) に沿って、誤りセルを
   `color_to_ojama` / `ojama_to_other` / `empty_confusion` / `color_confusion`
   の4分類に分けて OFF/(b)/(c) それぞれで集計し、色→おじゃま誤検出が
   ゲートでどれだけ減ったかを追跡する。

## 6. 第3系統追加 (2026-08-05): バーストガード v2 (Stage1)
`docs/BURST_GUARD_DESIGN_2026-08-05.md` §7.1 に対応する第3の比較系統
`v2` (`enable_burst_guard_v2`、Schmitt trigger視覚トリガー + ハード凍結) を
追加する。31動画再認識の裏走行結果を `data/verify/burst_guard_2026-08-05/
on_v2/` に着弾させる想定 (未着弾動画は既存 (c) と同様に STAGE_UNAVAILABLE
として明示スキップし、着弾分のみ集計する)。`--v2-npz-dir` で上書き可能
(既定値は未指定時と同一、backwards compat)。

同時に **layer別集計** (burst layer = row1-3 / smoke layer = row4-12) を
追加する。Stage1 は burst layer (視覚トリガーの守備範囲) のみ改善する見込み
で、smoke layer (お邪魔着弾由来、Stage2 で対応予定) は不変が正直な期待値
(`feedback_overfitting_awareness_2026-08-04`: 都合の良い数字だけ見せない)。
`count_cell_errors`/`count_error_categories` に `rows: frozenset[int]|None`
を追加し (既定 None = 全行、既存呼び出しは bit-identical)、`StageResult`/
`CompareResult` に layer別カウントを追加した (既存フィールドは変更なし、
新規フィールドは全て default 付きで末尾追加、backwards compat)。

## 検収基準

OFF 側の誤りセル合計が **93** (batch1=47 / batch2=46) に一致すること
(全行、layer 分割前の合計。この判定基準自体は変更しない)。
一致しなければ突合ロジックが壊れている扱いとし、(b)/(c)/(v2) の効果集計は
出力しない (`feedback_overfitting_awareness_2026-08-04`: 数字が合うまで
採否集計を出さない)。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.measure_effect_gate_c_2026-08-04
    PYTHONPATH=. ./venv/bin/python -m scripts.measure_effect_gate_c_2026-08-04 \\
        --v2-npz-dir data/verify/burst_guard_2026-08-05/on_v2
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import (  # noqa: E402
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
)
from src.board_state_machine import EFFECT_GATE_TOP_ROWS  # noqa: E402

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

LABEL_UNKNOWN_CHAR: str = "U"

# batch1/batch2 ラベルシートの場所 (CLAUDE.md: 動画追加時のティア filter は
# 既に build_full_board_label_sheet.py 側で完了済みのデータを使う)。
BATCH1_DIR: Path = Path("data/verify/full_board_label_sheet_2026-08-02")
BATCH2_DIR: Path = Path("data/verify/full_board_label_sheet_batch2_2026-08-03")

# アンカー = OFF 基準 npz (本番収集そのもの、再収集不要)。
ANCHOR_NPZ_DIR: Path = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
# (b) 時間ゲートのみ ON (2026-08-03 v2、batch1 の24動画のみ存在)。
NPZ_DIR_B: Path = Path("data/verify/effect_gate_2026-08-03_v2/on")
# (c) 4条件フルAND (2026-08-04、31動画対象で裏走行中)。
NPZ_DIR_C: Path = Path("data/verify/effect_gate_2026-08-04_c/on_full")
# (v2) バーストガード Stage1 (2026-08-05、31動画対象で裏走行中、着弾予定)。
NPZ_DIR_V2: Path = Path("data/verify/burst_guard_2026-08-05/on_v2")

# layer別集計 (2026-08-05、docs/BURST_GUARD_DESIGN_2026-08-05.md §7.1)。
# burst layer = 視覚トリガーの守備範囲 (既存 EFFECT_GATE_TOP_ROWS を再利用、
# 再定義しない)。smoke layer = それ以外の可視行 (row4-12、お邪魔着弾由来、
# Stage1 では不変が正直な期待値)。row0 は隠し段で常に U のため対象外。
LAYER_BURST_ROWS: "frozenset[int]" = EFFECT_GATE_TOP_ROWS
LAYER_SMOKE_ROWS: "frozenset[int]" = frozenset(range(4, BOARD_ROWS))
LAYER_FULL: str = "full"
LAYER_BURST: str = "burst"
LAYER_SMOKE: str = "smoke"

# アンカー行を自身の npz 内で一意確定するための時刻許容誤差 (秒)。
# ラベルの t_sec は小数第1位に丸められているため、ごく小さい誤差で十分
# (v3 と同一の値・根拠)。
ANCHOR_LOOKUP_TOLERANCE_SEC: float = 1.0
# (b)/(c) npz で frame_idx 完全一致が取れない場合の bit-exact フォールバック
# 探索窓 (秒、v3 と同一の値)。
ANCHOR_MATCH_WINDOW_SEC: float = 30.0

# 検収基準: バッチ別 + 合計の OFF 誤りセル数期待値。
EXPECTED_OFF_TOTAL_BATCH1: int = 47
EXPECTED_OFF_TOTAL_BATCH2: int = 46
EXPECTED_OFF_TOTAL: int = EXPECTED_OFF_TOTAL_BATCH1 + EXPECTED_OFF_TOTAL_BATCH2

# 盤面単位の内訳 CSV 出力先。
OUTPUT_CSV_PATH: Path = Path(
    "data/verify/effect_gate_2026-08-04_c/measure_c_board_breakdown.csv"
)

# 誤りセル分類ラベル (罪の序列: project_full_board_error_taxonomy_2026-08-02)。
CATEGORY_COLOR_TO_OJAMA: str = "color_to_ojama"   # 罪1位: 色→おじゃま誤検出
CATEGORY_OJAMA_TO_OTHER: str = "ojama_to_other"   # おじゃま誤認 (逆方向)
CATEGORY_EMPTY_CONFUSION: str = "empty_confusion"  # 空との混同
CATEGORY_COLOR_CONFUSION: str = "color_confusion"  # 色同士の混同
ERROR_CATEGORIES: tuple[str, ...] = (
    CATEGORY_COLOR_TO_OJAMA, CATEGORY_OJAMA_TO_OTHER,
    CATEGORY_EMPTY_CONFUSION, CATEGORY_COLOR_CONFUSION,
)

# npz 未存在時の状態ラベル (b/c 系列共通)。
STAGE_MATCHED: str = "matched"        # frame_idx 完全一致行が見つかった
STAGE_UNAVAILABLE: str = "unavailable"  # npz ファイル自体が存在しない (N/A)
STAGE_NO_MATCH: str = "no_match"      # npz は存在するが該当 frame_idx がない (要調査)


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class LabelSample:
    """1 件のラベル済み満杯盤面サンプル (batch タグ付き)。"""

    batch: str            # "batch1" / "batch2"
    video_stem: str        # "c10"
    t_sec: float
    side: str               # "1P" / "2P"
    status: str             # "ok" / "fixed"
    correct_grid: "np.ndarray"  # (13, 6) int、U は COLOR_UNKNOWN
    game_idx: "int | None"
    anchor_recognized_grid: "np.ndarray | None"


@dataclass
class StageResult:
    """(b)/(c)/(v2) npz との突合結果 (1 サンプル分)。"""

    status: str  # STAGE_MATCHED / STAGE_UNAVAILABLE / STAGE_NO_MATCH
    errors: "int | None"
    categories: "dict[str, int] | None"
    # 一致した方式 ("frame_idx" / "bit_exact_fallback")。status!=matched では None。
    matched_via: "str | None" = None
    # layer別誤りセル数 (2026-08-05 追加、末尾 default 付きで backwards compat)。
    errors_burst: "int | None" = None  # burst layer (row1-3)
    errors_smoke: "int | None" = None  # smoke layer (row4-12)


@dataclass
class CompareResult:
    """アンカー(=OFF)突合 + (b)/(c)/(v2) 誤りセル数比較結果 (1 サンプル分)。"""

    sample: LabelSample
    anchor_found: bool
    off_errors: "int | None"
    off_categories: "dict[str, int] | None"
    verified_t_sec: "float | None"
    b: StageResult
    c: StageResult
    # (v2) バーストガード Stage1 (2026-08-05 追加、末尾 default 付きで backwards compat)。
    v2: StageResult = field(
        default_factory=lambda: StageResult(STAGE_UNAVAILABLE, None, None)
    )
    off_errors_burst: "int | None" = None
    off_errors_smoke: "int | None" = None


# =============================================================================
# 1. ラベル読み込み (batch1 + batch2 統合)
# =============================================================================


def decode_grid_string(s: str) -> "np.ndarray":
    """"UUU00U/992009/..." 形式を (13, 6) int の numpy 配列に変換する。"""
    rows = s.split("/")
    assert len(rows) == BOARD_ROWS, f"行数不正: {len(rows)} != {BOARD_ROWS}"
    grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int64)
    for r, row_str in enumerate(rows):
        assert len(row_str) == BOARD_COLS, f"列数不正 row={r}: {row_str!r}"
        for c, ch in enumerate(row_str):
            grid[r, c] = COLOR_UNKNOWN if ch == LABEL_UNKNOWN_CHAR else int(ch)
    return grid


def _load_recognized_grid_lookup(
    sheet_csv: Path,
) -> "dict[tuple[str, str, str], tuple[str, int]]":
    """(video_id, t_sec文字列, side) -> (recognized_grid, game_idx) の辞書。"""
    lookup: dict[tuple[str, str, str], tuple[str, int]] = {}
    with sheet_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["video_id"], row["t_sec"], row["side"])
            lookup[key] = (row["recognized_grid"], int(row["game_idx"]))
    return lookup


def load_label_samples_for_batch(
    batch: str, result_csv: Path, sheet_csv: Path,
) -> list[LabelSample]:
    """1 batch 分の labeling_result.csv から ok/fixed 行のみを読み込む。"""
    recognized_lookup = _load_recognized_grid_lookup(sheet_csv)
    samples: list[LabelSample] = []
    with result_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            status = row["status"]
            if status not in ("ok", "fixed"):
                continue
            key = (row["video_id"], row["t_sec"], row["side"])
            recognized_str, game_idx = recognized_lookup.get(key, ("", None))
            grid_str = row["correct_grid"] if status == "fixed" else recognized_str
            if not grid_str or not recognized_str:
                continue
            samples.append(LabelSample(
                batch=batch,
                video_stem=row["video_id"].replace("video_", ""),
                t_sec=float(row["t_sec"]),
                side=row["side"],
                status=status,
                correct_grid=decode_grid_string(grid_str),
                game_idx=game_idx,
                anchor_recognized_grid=decode_grid_string(recognized_str),
            ))
    return samples


def load_all_samples() -> list[LabelSample]:
    """batch1 + batch2 の ok/fixed サンプルを統合して返す (計55件想定)。"""
    batch1 = load_label_samples_for_batch(
        "batch1", BATCH1_DIR / "labeling_result.csv", BATCH1_DIR / "labeling_sheet.csv",
    )
    batch2 = load_label_samples_for_batch(
        "batch2", BATCH2_DIR / "labeling_result.csv", BATCH2_DIR / "labeling_sheet.csv",
    )
    return batch1 + batch2


# =============================================================================
# 2. npz index + アンカー突合 (v3 と同一ロジック)
# =============================================================================


@dataclass
class _NpzIndex:
    """1 npz の高速検索用インデックス。"""

    t_secs: "np.ndarray"
    sides: "np.ndarray"
    grids: "np.ndarray"
    frame_idxs: "np.ndarray"
    game_idxs: "np.ndarray"


def _load_npz_index(npz_path: Path) -> "_NpzIndex | None":
    """npz ファイルが存在しない場合は None (= 未着弾/対象外を表す)。"""
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    return _NpzIndex(
        t_secs=data["t_sec"].astype(np.float64),
        sides=data["side"],
        grids=data["grids"],
        frame_idxs=data["frame_idx"].astype(np.int64),
        game_idxs=data["game_idx"].astype(np.int64),
    )


@dataclass
class _AnchorRow:
    """アンカー (= OFF 基準 npz) の該当行。"""

    grid: "np.ndarray"
    frame_idx: int
    t_sec: float


def _lookup_anchor_row(
    idx: "_NpzIndex | None", side: str, t_sec: float, game_idx: "int | None",
    expected_grid: "np.ndarray | None",
) -> "_AnchorRow | None":
    """アンカー npz から (side, game_idx, t_sec近傍) で該当行を一意に確定する。

    labeling_sheet.csv の recognized_grid (= アンカーの文字列表現そのもの) と
    bit 一致する行を優先して探し、丸め誤差による隣接行の誤掴みを防ぐ
    (v3 と同一ロジック、根拠は本ファイル docstring 参照)。
    """
    if idx is None:
        return None
    mask = idx.sides == side
    if game_idx is not None:
        mask = mask & (idx.game_idxs == game_idx)
    if not mask.any():
        return None
    cand = np.where(mask)[0]
    diffs = np.abs(idx.t_secs[cand] - t_sec)
    within = cand[diffs <= ANCHOR_LOOKUP_TOLERANCE_SEC]
    if len(within) == 0:
        return None
    if expected_grid is not None:
        exact = [i for i in within if np.array_equal(idx.grids[i], expected_grid)]
        if exact:
            best = min(exact, key=lambda i: abs(idx.t_secs[i] - t_sec))
            return _AnchorRow(
                grid=idx.grids[best], frame_idx=int(idx.frame_idxs[best]),
                t_sec=float(idx.t_secs[best]),
            )
    # フォールバック: bit 一致行が見つからない場合のみ最近傍時刻を採用。
    diffs_within = np.abs(idx.t_secs[within] - t_sec)
    best = within[int(np.argmin(diffs_within))]
    return _AnchorRow(
        grid=idx.grids[best], frame_idx=int(idx.frame_idxs[best]),
        t_sec=float(idx.t_secs[best]),
    )


def _find_by_frame_idx_exact(
    idx: "_NpzIndex", side: str, frame_idx: int,
) -> "tuple[np.ndarray, float] | None":
    """side 一致 + frame_idx 完全一致の行を探す (認識は決定論的なので同一
    動画・同一 frame なら常に同じ frame_idx になるはず)。"""
    mask = (idx.sides == side) & (idx.frame_idxs == frame_idx)
    cand = np.where(mask)[0]
    if len(cand) == 0:
        return None
    best = int(cand[0])
    return idx.grids[best], float(idx.t_secs[best])


def _find_bit_exact_match(
    idx: "_NpzIndex", side: str, anchor_grid: "np.ndarray",
    center_t_sec: float, window_sec: float,
) -> "tuple[np.ndarray, float] | None":
    """side 一致 + anchor_grid と bit 一致する行を、center_t_sec に最も近い順に探す。

    frame_idx 完全一致検索のフォールバック (v3 と同一)。dedup 間引き
    タイミングの差で frame_idx がわずかにずれても、内容が「無変化」の
    区間なら回収できる (内容が変わった真の改善/劣化ケースは bit 一致しない
    ため本フォールバックでは検出されず frame_idx 一致側に委ねられる、
    本ファイル docstring 差分4 参照)。
    """
    mask = (idx.sides == side) & (np.abs(idx.t_secs - center_t_sec) <= window_sec)
    cand_idx = np.where(mask)[0]
    if len(cand_idx) == 0:
        return None
    exact = [i for i in cand_idx if np.array_equal(idx.grids[i], anchor_grid)]
    if not exact:
        return None
    best = min(exact, key=lambda i: abs(idx.t_secs[i] - center_t_sec))
    return idx.grids[best], float(idx.t_secs[best])


# =============================================================================
# 3. 誤りセル計算 + 罪の序列分類
# =============================================================================


def _row_mask(rows: "frozenset[int] | None") -> "np.ndarray | None":
    """rows (対象行集合) から (BOARD_ROWS, BOARD_COLS) 形の bool mask を作る。

    None なら None を返す (呼び出し側は「全行対象」として扱う、backwards compat)。
    """
    if rows is None:
        return None
    mask = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=bool)
    mask[sorted(rows), :] = True
    return mask


def count_cell_errors(
    predicted: "np.ndarray", correct: "np.ndarray",
    rows: "frozenset[int] | None" = None,
) -> int:
    """正解ラベルが U (不明) でないセルのみを対象に誤りセル数を数える。

    rows: 2026-08-05 追加。指定した行集合のみを対象にする (layer別集計用)。
        None (既定) なら全行対象 (既存呼び出しと bit-identical、backwards compat)。
    """
    known_mask = correct != COLOR_UNKNOWN
    row_mask = _row_mask(rows)
    if row_mask is not None:
        known_mask = known_mask & row_mask
    return int(np.count_nonzero(
        (predicted.astype(np.int64) != correct.astype(np.int64)) & known_mask
    ))


def classify_error_category(predicted: int, correct: int) -> "str | None":
    """1 セルの誤り分類 (罪の序列: project_full_board_error_taxonomy_2026-08-02)。

    predicted == correct のセルは None (誤りでない) を返す。
    """
    if predicted == correct:
        return None
    if predicted == COLOR_OJAMA:
        return CATEGORY_COLOR_TO_OJAMA  # 色 (または空) → おじゃま誤検出
    if correct == COLOR_OJAMA:
        return CATEGORY_OJAMA_TO_OTHER  # おじゃま → 他 誤認
    if predicted == COLOR_EMPTY or correct == COLOR_EMPTY:
        return CATEGORY_EMPTY_CONFUSION
    return CATEGORY_COLOR_CONFUSION


def count_error_categories(
    predicted: "np.ndarray", correct: "np.ndarray",
    rows: "frozenset[int] | None" = None,
) -> "dict[str, int]":
    """盤面 (または指定 rows のみ) の誤りセルを罪の序列カテゴリ別に集計する。

    rows: 2026-08-05 追加。None (既定) なら全行対象 (bit-identical、backwards compat)。
    """
    counts: dict[str, int] = {c: 0 for c in ERROR_CATEGORIES}
    target_rows = range(BOARD_ROWS) if rows is None else sorted(rows)
    for r in target_rows:
        for c in range(BOARD_COLS):
            cv = int(correct[r, c])
            if cv == COLOR_UNKNOWN:
                continue
            cat = classify_error_category(int(predicted[r, c]), cv)
            if cat is not None:
                counts[cat] += 1
    return counts


# =============================================================================
# 4. メイン比較処理
# =============================================================================


def compute_stage_result(
    idx: "_NpzIndex | None", side: str, anchor: "_AnchorRow", correct_grid: "np.ndarray",
) -> StageResult:
    """(b)/(c)/(v2) npz との突合を行う (v3 と同一の2段構成)。

    ① frame_idx 完全一致 → ② bit-exact 時刻窓フォールバック。両方失敗なら
    STAGE_NO_MATCH として明示的に区別する (fail-silent 回避)。
    layer別誤りセル数 (burst/smoke) も同時に計算する (2026-08-05 追加)。
    """
    if idx is None:
        return StageResult(STAGE_UNAVAILABLE, None, None)
    match = _find_by_frame_idx_exact(idx, side, anchor.frame_idx)
    matched_via = "frame_idx"
    if match is None:
        match = _find_bit_exact_match(
            idx, side, anchor.grid, anchor.t_sec, ANCHOR_MATCH_WINDOW_SEC,
        )
        matched_via = "bit_exact_fallback"
    if match is None:
        return StageResult(STAGE_NO_MATCH, None, None)
    grid, _t = match
    errors = count_cell_errors(grid, correct_grid)
    categories = count_error_categories(grid, correct_grid)
    errors_burst = count_cell_errors(grid, correct_grid, rows=LAYER_BURST_ROWS)
    errors_smoke = count_cell_errors(grid, correct_grid, rows=LAYER_SMOKE_ROWS)
    return StageResult(
        STAGE_MATCHED, errors, categories, matched_via, errors_burst, errors_smoke,
    )


def compare_one_sample(
    s: LabelSample,
    anchor_idx: "_NpzIndex | None",
    b_idx: "_NpzIndex | None",
    c_idx: "_NpzIndex | None",
    v2_idx: "_NpzIndex | None" = None,
) -> CompareResult:
    """1 サンプル分の OFF(=アンカー)/(b)/(c)/(v2) 誤りセル数を確定する。"""
    anchor = _lookup_anchor_row(
        anchor_idx, s.side, s.t_sec, s.game_idx, s.anchor_recognized_grid,
    )
    unavailable = StageResult(STAGE_UNAVAILABLE, None, None)
    if anchor is None:
        return CompareResult(
            s, False, None, None, None, unavailable, unavailable, unavailable,
        )

    off_errors = count_cell_errors(anchor.grid, s.correct_grid)
    off_categories = count_error_categories(anchor.grid, s.correct_grid)
    off_errors_burst = count_cell_errors(anchor.grid, s.correct_grid, rows=LAYER_BURST_ROWS)
    off_errors_smoke = count_cell_errors(anchor.grid, s.correct_grid, rows=LAYER_SMOKE_ROWS)
    b_result = compute_stage_result(b_idx, s.side, anchor, s.correct_grid)
    c_result = compute_stage_result(c_idx, s.side, anchor, s.correct_grid)
    v2_result = compute_stage_result(v2_idx, s.side, anchor, s.correct_grid)
    return CompareResult(
        s, True, off_errors, off_categories, anchor.t_sec, b_result, c_result,
        v2_result, off_errors_burst, off_errors_smoke,
    )


def compare_all_samples(
    samples: list[LabelSample], v2_dir: Path = NPZ_DIR_V2,
) -> list[CompareResult]:
    """全サンプルについてアンカー突合 + OFF/(b)/(c)/(v2) 誤りセル数を計算する。

    v2_dir: 2026-08-05 追加。既定値 (NPZ_DIR_V2) 未変更なら既存呼び出しと
        bit-identical (backwards compat)。
    """
    anchor_cache: dict[str, "_NpzIndex | None"] = {}
    b_cache: dict[str, "_NpzIndex | None"] = {}
    c_cache: dict[str, "_NpzIndex | None"] = {}
    v2_cache: dict[str, "_NpzIndex | None"] = {}
    results: list[CompareResult] = []
    for s in samples:
        if s.video_stem not in anchor_cache:
            anchor_cache[s.video_stem] = _load_npz_index(
                ANCHOR_NPZ_DIR / f"{s.video_stem}.npz"
            )
            b_cache[s.video_stem] = _load_npz_index(NPZ_DIR_B / f"{s.video_stem}.npz")
            c_cache[s.video_stem] = _load_npz_index(NPZ_DIR_C / f"{s.video_stem}.npz")
            v2_cache[s.video_stem] = _load_npz_index(v2_dir / f"{s.video_stem}.npz")
        results.append(compare_one_sample(
            s, anchor_cache[s.video_stem], b_cache[s.video_stem], c_cache[s.video_stem],
            v2_cache[s.video_stem],
        ))
    return results


# =============================================================================
# 5. 検収 (OFF=93 の再現確認)
# =============================================================================


def off_total_by_batch(results: list[CompareResult]) -> "dict[str, int]":
    """batch 別の OFF 誤りセル合計 (アンカー突合成功分のみ)。"""
    totals: dict[str, int] = defaultdict(int)
    for r in results:
        if r.off_errors is not None:
            totals[r.sample.batch] += r.off_errors
    return dict(totals)


def verify_off_baseline(results: list[CompareResult]) -> tuple[bool, str]:
    """OFF 誤りセル合計が batch1=47/batch2=46/合計93 と一致するか検収する。"""
    totals = off_total_by_batch(results)
    b1 = totals.get("batch1", 0)
    b2 = totals.get("batch2", 0)
    total = b1 + b2
    ok = (
        b1 == EXPECTED_OFF_TOTAL_BATCH1
        and b2 == EXPECTED_OFF_TOTAL_BATCH2
        and total == EXPECTED_OFF_TOTAL
    )
    msg = (
        f"[検収] OFF誤りセル合計: batch1={b1}(期待{EXPECTED_OFF_TOTAL_BATCH1}) "
        f"batch2={b2}(期待{EXPECTED_OFF_TOTAL_BATCH2}) "
        f"合計={total}(期待{EXPECTED_OFF_TOTAL}) → {'合格' if ok else '★不合格★'}"
    )
    return ok, msg


# =============================================================================
# 6. 集計レポート (層別必須: batch × side)
# =============================================================================


_STAGE_LABELS: "tuple[str, ...]" = ("b", "c", "v2")
_LAYER_LABELS: "dict[str, str]" = {
    LAYER_FULL: "全体",
    LAYER_BURST: "burstレイヤー(row1-3)",
    LAYER_SMOKE: "smokeレイヤー(row4-12)",
}


def _stage_result_for(r: CompareResult, stage: str) -> StageResult:
    """stage 名 ("b"/"c"/"v2") から対応する StageResult を取り出す。"""
    return {"b": r.b, "c": r.c, "v2": r.v2}[stage]


def _layer_value(sr: StageResult, layer: str) -> "int | None":
    """StageResult から layer 別誤りセル数を取り出す (LAYER_FULL は全体)。"""
    if layer == LAYER_BURST:
        return sr.errors_burst
    if layer == LAYER_SMOKE:
        return sr.errors_smoke
    return sr.errors


def _off_layer_value(r: CompareResult, layer: str) -> "int | None":
    """CompareResult から OFF の layer 別誤りセル数を取り出す。"""
    if layer == LAYER_BURST:
        return r.off_errors_burst
    if layer == LAYER_SMOKE:
        return r.off_errors_smoke
    return r.off_errors


def _stage_error_sum(
    results: list[CompareResult], stage: str, layer: str = LAYER_FULL,
) -> "tuple[int, int, int]":
    """stage ("b"/"c"/"v2") の (matched件数, no_match件数, 誤りセル合計) を返す。

    layer: 2026-08-05 追加。LAYER_FULL (既定) なら全行合計 (bit-identical)。
    """
    matched = 0
    no_match = 0
    total_errors = 0
    for r in results:
        sr = _stage_result_for(r, stage)
        if sr.status == STAGE_MATCHED:
            matched += 1
            total_errors += _layer_value(sr, layer) or 0
        elif sr.status == STAGE_NO_MATCH:
            no_match += 1
    return matched, no_match, total_errors


def _stratified_row(
    batch: str, side: str, grp: list[CompareResult], layer: str,
) -> str:
    """batch × side 1グループ分の層別集計行 (OFF/(b)/(c)/(v2))。"""
    off_sum = sum((_off_layer_value(r, layer) or 0) for r in grp)
    cells = []
    for stage in _STAGE_LABELS:
        m, nm, err = _stage_error_sum(grp, stage, layer)
        na = sum(1 for r in grp if _stage_result_for(r, stage).status == STAGE_UNAVAILABLE)
        cells.append(f"{f'{m}/{na}/{nm}':>16} {err:>6}")
    return f"{batch:8} {side:4} {len(grp):>3} {off_sum:>5} " + " ".join(cells)


def build_stratified_table(
    results: list[CompareResult], layer: str = LAYER_FULL,
) -> str:
    """batch × side 層別の OFF/(b)/(c)/(v2) 誤りセル数表 (feedback_stratify_before_pooling)。

    layer: LAYER_FULL (既定、全行) / LAYER_BURST (row1-3) / LAYER_SMOKE (row4-12)。
    """
    lines = [f"--- 層別集計 (batch × side) [{_LAYER_LABELS[layer]}] ---"]
    groups: "dict[tuple[str, str], list[CompareResult]]" = defaultdict(list)
    for r in results:
        groups[(r.sample.batch, r.sample.side)].append(r)
    header = (
        f"{'batch':8} {'side':4} {'n':>3} {'OFF':>5} "
        + " ".join(f"{f'{s}(m/na/nm)':>16} {s + '_err':>6}" for s in _STAGE_LABELS)
    )
    lines.append(header)
    for (batch, side), grp in sorted(groups.items()):
        lines.append(_stratified_row(batch, side, grp, layer))
    return "\n".join(lines)


def build_category_reduction_report(results: list[CompareResult]) -> str:
    """罪の序列カテゴリ別の OFF→(b)→(c)→(v2) 誤りセル数遷移 (色→9誤検出の減少が主目的)。"""
    lines = ["--- 誤り分類別 OFF→(b)→(c)→(v2) 遷移 (色→おじゃま誤検出=最重要) ---"]
    off_totals: "Counter[str]" = Counter()
    stage_totals: "dict[str, Counter[str]]" = {s: Counter() for s in _STAGE_LABELS}
    for r in results:
        if r.off_categories is not None:
            off_totals.update(r.off_categories)
        for stage in _STAGE_LABELS:
            sr = _stage_result_for(r, stage)
            if sr.categories is not None:
                stage_totals[stage].update(sr.categories)
    header = f"{'category':20} {'OFF':>6}" + "".join(
        f" {s + '(matched分)':>14}" for s in _STAGE_LABELS
    )
    lines.append(header)
    for cat in ERROR_CATEGORIES:
        row = f"{cat:20} {off_totals[cat]:>6}" + "".join(
            f" {stage_totals[s][cat]:>14}" for s in _STAGE_LABELS
        )
        lines.append(row)
    return "\n".join(lines)


def build_anomaly_report(results: list[CompareResult]) -> str:
    """明示報告すべき異常 (アンカー突合失敗 / no_match) の一覧。"""
    lines = ["--- 異常一覧 (fail-silent 回避のため必ず確認) ---"]
    n_anchor_fail = sum(1 for r in results if not r.anchor_found)
    no_match_counts = {
        s: sum(1 for r in results if _stage_result_for(r, s).status == STAGE_NO_MATCH)
        for s in _STAGE_LABELS
    }
    fallback_counts = {
        s: sum(1 for r in results if _stage_result_for(r, s).matched_via == "bit_exact_fallback")
        for s in _STAGE_LABELS
    }
    lines.append(
        f"アンカー突合失敗: {n_anchor_fail} 件 / "
        + " / ".join(f"({s}) no_match: {no_match_counts[s]} 件" for s in _STAGE_LABELS)
    )
    lines.append(
        "bit-exact フォールバック使用 (frame_idx 不一致だが内容無変化で回収): "
        + " / ".join(f"({s}) {fallback_counts[s]} 件" for s in _STAGE_LABELS)
    )
    for r in results:
        s = r.sample
        tags = [] if r.anchor_found else ["anchor_fail"]
        tags += [
            f"{stage}_no_match" for stage in _STAGE_LABELS
            if _stage_result_for(r, stage).status == STAGE_NO_MATCH
        ]
        if tags:
            lines.append(
                f"  {s.batch} video_{s.video_stem} t={s.t_sec:.1f} {s.side}: "
                f"{','.join(tags)}"
            )
    return "\n".join(lines)


def build_landing_report(results: list[CompareResult]) -> str:
    """(c)/(v2) npz の着弾状況 (何動画分が集計対象になっているか)。"""
    all_videos = sorted({r.sample.video_stem for r in results})
    lines = []
    for stage in ("c", "v2"):
        landed = sorted({
            r.sample.video_stem for r in results
            if _stage_result_for(r, stage).status != STAGE_UNAVAILABLE
        })
        lines.append(
            f"[({stage}) 着弾状況] {len(landed)}/{len(all_videos)} 動画着弾済み "
            f"(未着弾: {sorted(set(all_videos) - set(landed))})"
        )
    return "\n".join(lines)


# =============================================================================
# 7. 盤面単位の内訳 CSV
# =============================================================================


def _stage_cell_for_csv(sr: StageResult) -> "tuple[str, str]":
    """CSV 出力用に StageResult を (誤りセル数文字列, 色→9誤検出数文字列) に変換。"""
    if sr.status != STAGE_MATCHED:
        return sr.status, sr.status
    err_str = str(sr.errors)
    cat_str = str((sr.categories or {}).get(CATEGORY_COLOR_TO_OJAMA, 0))
    return err_str, cat_str


def _int_or_na(value: "int | None") -> str:
    """None を "N/A" 文字列にする (CSV セル用の小ヘルパー)。"""
    return "N/A" if value is None else str(value)


def write_breakdown_csv(results: list[CompareResult], out_path: Path) -> None:
    """盤面単位の内訳 CSV を出力する。

    既存列 (off_err/off_color_to_ojama/b_err/b_color_to_ojama/c_err/
    c_color_to_ojama) は順序・内容とも変更しない (backwards compat)。
    2026-08-05: v2列 + layer別列 (burst=row1-3/smoke=row4-12) を末尾に追加。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch", "video", "side", "t_sec", "status",
        "off_err", "off_color_to_ojama",
        "b_err", "b_color_to_ojama",
        "c_err", "c_color_to_ojama",
        "v2_err", "v2_color_to_ojama",
        "off_err_burst", "off_err_smoke",
        "v2_err_burst", "v2_err_smoke",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(_breakdown_row(r))


def _breakdown_row(r: CompareResult) -> "dict[str, str]":
    """1サンプル分の CSV 行を組み立てる (50行規約回避のため分離)。"""
    s = r.sample
    off_err = "anchor_fail" if r.off_errors is None else str(r.off_errors)
    off_cat = (
        "anchor_fail" if r.off_categories is None
        else str(r.off_categories.get(CATEGORY_COLOR_TO_OJAMA, 0))
    )
    b_err, b_cat = _stage_cell_for_csv(r.b)
    c_err, c_cat = _stage_cell_for_csv(r.c)
    v2_err, v2_cat = _stage_cell_for_csv(r.v2)
    return {
        "batch": s.batch, "video": s.video_stem, "side": s.side,
        "t_sec": f"{s.t_sec:.1f}", "status": s.status,
        "off_err": off_err, "off_color_to_ojama": off_cat,
        "b_err": b_err, "b_color_to_ojama": b_cat,
        "c_err": c_err, "c_color_to_ojama": c_cat,
        "v2_err": v2_err, "v2_color_to_ojama": v2_cat,
        "off_err_burst": _int_or_na(r.off_errors_burst),
        "off_err_smoke": _int_or_na(r.off_errors_smoke),
        "v2_err_burst": _int_or_na(r.v2.errors_burst),
        "v2_err_smoke": _int_or_na(r.v2.errors_smoke),
    }


# =============================================================================
# 8. main
# =============================================================================


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    """CLI引数を解析する (2026-08-05 追加、未指定時は既存挙動と bit-identical)。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2-npz-dir", type=Path, default=NPZ_DIR_V2,
        help=f"バーストガード v2 npz ディレクトリ (既定: {NPZ_DIR_V2})",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    samples = load_all_samples()
    n_batch1 = sum(1 for s in samples if s.batch == "batch1")
    n_batch2 = sum(1 for s in samples if s.batch == "batch2")
    print(f"[1/3] ラベル読込: {len(samples)} 件 (batch1={n_batch1} / batch2={n_batch2})")

    results = compare_all_samples(samples, v2_dir=args.v2_npz_dir)
    print("[2/3] アンカー突合 + OFF/(b)/(c)/(v2) 誤りセル数比較")
    print(build_anomaly_report(results))
    print()

    ok, verify_msg = verify_off_baseline(results)
    print(verify_msg)
    write_breakdown_csv(results, OUTPUT_CSV_PATH)
    print(f"[出力] 盤面単位内訳 CSV: {OUTPUT_CSV_PATH}")

    if not ok:
        print(
            "\n★★★ OFF 検収不合格のため (b)/(c)/(v2) 効果集計は出力しません "
            "(feedback_overfitting_awareness_2026-08-04: 数字が合うまで採否集計を出さない) ★★★"
        )
        return

    print("\n[3/3] (b)/(c)/(v2) 効果集計 (層別必須)")
    print(build_landing_report(results))
    print()
    # docs/BURST_GUARD_DESIGN_2026-08-05.md §7.1: burst/smoke layer別に必ず分ける。
    # Stage1 は burst layer のみ改善見込み、smoke layer は不変が正直な期待値。
    for layer in (LAYER_FULL, LAYER_BURST, LAYER_SMOKE):
        print(build_stratified_table(results, layer))
        print()
    print(build_category_reduction_report(results))


if __name__ == "__main__":
    main()
