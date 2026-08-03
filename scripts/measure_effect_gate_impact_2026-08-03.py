"""エフェクト時間ゲート (enable_effect_gate) 効果測定: 誤りセル数 OFF vs ON 比較 (2026-08-03, v2)。

満杯盤面 47 セル誤りの真因確定 (memory `project_full_board_error_taxonomy_2026-08-02`)
を受けて実装したエフェクト時間ゲートの効果を、user 正解ラベル
(data/verify/full_board_label_sheet_2026-08-02/labeling_result.csv) に対する
セル誤り数の減少で測定する。

## v1 からの修正 (検収不能だった3件の矛盾への対処)
1. **nearest t_sec 突合の破棄**: 「ok」ラベル (誤り0のはず) で非ゼロ誤りが出た
   根本原因は認識の非決定性ではなく、**再収集の pipeline 設定がラベル元と
   異なっていた**こと (--with-next 抜け → NextDetector 不在で着地色推論経路が
   変わり confirmed_board の値自体が変わっていた)。scripts/_run_effect_gate_measure_v2_2026-08-03.sh
   で --with-next を追加し設定を完全一致させた上で、**アンカー方式**
   (ラベル元 npz の該当行 = アンカー、OFF 再収集 npz でアンカーと grid が
   bit 一致する行を探して対応フレームを確定する) に切替えた。認識は
   決定論的 (Step0 で bit 一致実証済み) なので、設定が一致すれば
   bit-exact match が必ず見つかるはずであり、見つからない場合は
   突合の異常として明示的に報告する (fail-silent 回避)。
2. **N/A 5件**: 突合窓を ANCHOR_MATCH_WINDOW_SEC (30秒) に拡大。
3. **U セル除外**: correct_grid が 'U' (未検証) のセルは分母から除外
   (既存ロジックのまま、検証の結果これで OFF=47 が再現することを確認済み)。

## 検収基準
OFF 側の誤りセル合計が 47 (fixed 12 枚の既知誤り) に一致すること。
一致しなければ突合がまだ壊れている。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts.measure_effect_gate_impact_2026-08-03
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_UNKNOWN  # noqa: E402

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

LABEL_RESULT_CSV: Path = Path(
    "data/verify/full_board_label_sheet_2026-08-02/labeling_result.csv"
)
LABEL_SHEET_CSV: Path = Path(
    "data/verify/full_board_label_sheet_2026-08-02/labeling_sheet.csv"
)
# ラベル元 npz (build_full_board_label_sheet.py が参照した npz、アンカー取得用)。
ANCHOR_NPZ_DIR: Path = Path("data/indicators_v2/boards_lean_regen_2026-07-31")
# v2 再収集 npz (--with-next 追加修正版)。
NPZ_ROOT: Path = Path("data/verify/effect_gate_2026-08-03_v2")
MODE_OFF: str = "off"
MODE_ON: str = "on"

LABEL_UNKNOWN_CHAR: str = "U"
# アンカー (ラベル元 npz) 検索時の時刻許容誤差 (秒)。ラベルの t_sec は
# アンカー npz の値を小数第1位に丸めたものなので、ごく小さい誤差で十分。
ANCHOR_LOOKUP_TOLERANCE_SEC: float = 1.0
# OFF 再収集 npz でアンカーと bit 一致する行を探す時刻窓 (秒)。
# v1 で --with-next 抜けにより発生した「N/A 5件」は本質的には設定違いが原因
# だったが、窓を広めに取ることで境界ずれ耐性も持たせる。
ANCHOR_MATCH_WINDOW_SEC: float = 30.0
# OFF で確定した「検証済み実時刻」から ON の対応行を探す時刻窓 (秒)。
# エフェクトゲートは cell 反映を最大 EFFECT_PERSIST_SEC (既定0.4秒) 遅らせる
# だけなので、対応する ON snapshot は OFF とほぼ同時刻にあるはず。
ON_LOOKUP_WINDOW_SEC: float = 5.0
# 検収基準: OFF 側の誤りセル合計 (fixed 12枚の既知誤り)。
EXPECTED_OFF_TOTAL_ERRORS: int = 47


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class LabelSample:
    """1 件のラベル済み満杯盤面サンプル。"""

    video_stem: str      # "c10"
    t_sec: float
    side: str            # "1P" / "2P"
    status: str          # "ok" / "fixed"
    correct_grid: "np.ndarray"  # (13, 6) int、U は COLOR_UNKNOWN


@dataclass
class CompareResult:
    """アンカー突合による OFF/ON 誤りセル数比較結果 (1 サンプル分)。"""

    sample: LabelSample
    anchor_found: bool
    off_errors: "int | None"     # None = アンカー一致行が OFF に見つからない (異常)
    on_errors: "int | None"      # None = 対応する ON 行が見つからない (異常)
    verified_t_sec: "float | None"  # OFF 側でアンカーが bit 一致した実 t_sec


# =============================================================================
# 1. ラベル読み込み
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
    csv_path: Path = LABEL_SHEET_CSV,
) -> "dict[tuple[str, str, str], str]":
    """(video_id, t_sec文字列, side) -> recognized_grid の辞書 (「ok」行の真値用)。"""
    lookup: dict[tuple[str, str, str], str] = {}
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lookup[(row["video_id"], row["t_sec"], row["side"])] = row["recognized_grid"]
    return lookup


def load_label_samples(csv_path: Path = LABEL_RESULT_CSV) -> list[LabelSample]:
    """labeling_result.csv から ok/fixed 行のみを読み込む。"""
    recognized_lookup = _load_recognized_grid_lookup()
    samples: list[LabelSample] = []
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            status = row["status"]
            if status not in ("ok", "fixed"):
                continue
            if status == "fixed":
                grid_str = row["correct_grid"]
            else:
                grid_str = recognized_lookup.get(
                    (row["video_id"], row["t_sec"], row["side"]), "",
                )
            if not grid_str:
                continue
            samples.append(LabelSample(
                video_stem=row["video_id"].replace("video_", ""),
                t_sec=float(row["t_sec"]),
                side=row["side"],
                status=status,
                correct_grid=decode_grid_string(grid_str),
            ))
    return samples


# =============================================================================
# 2. npz index + アンカー突合
# =============================================================================


@dataclass
class _NpzIndex:
    """1 npz の高速検索用インデックス。"""

    t_secs: "np.ndarray"
    sides: "np.ndarray"
    grids: "np.ndarray"


def _load_npz_index(npz_path: Path) -> "_NpzIndex | None":
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    return _NpzIndex(
        t_secs=data["t_sec"].astype(np.float64),
        sides=data["side"],
        grids=data["grids"],
    )


def _lookup_anchor_grid(
    idx: "_NpzIndex | None", side: str, t_sec: float,
) -> "np.ndarray | None":
    """ラベル元 npz から (side, t_sec) に最も近いアンカー grid を取得する。"""
    if idx is None:
        return None
    mask = idx.sides == side
    if not mask.any():
        return None
    diffs = np.abs(idx.t_secs[mask] - t_sec)
    best_i = int(np.argmin(diffs))
    if diffs[best_i] > ANCHOR_LOOKUP_TOLERANCE_SEC:
        return None
    return idx.grids[mask][best_i]


def _find_bit_exact_match(
    idx: "_NpzIndex", side: str, anchor_grid: "np.ndarray",
    center_t_sec: float, window_sec: float,
) -> "tuple[np.ndarray, float] | None":
    """side 一致 + anchor_grid と bit 一致する行を、center_t_sec に最も近い順に探す。

    複数の完全一致 (盤面が変化しない区間) がありうるため、時刻窓内に
    絞った上で center_t_sec に最も近いものを採用する。
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


def _find_nearest_snapshot(
    idx: "_NpzIndex", side: str, t_sec: float, window_sec: float,
) -> "np.ndarray | None":
    """side 一致で t_sec に最も近い行を時刻窓内から探す (bit 一致は問わない)。"""
    mask = (idx.sides == side) & (np.abs(idx.t_secs - t_sec) <= window_sec)
    cand_idx = np.where(mask)[0]
    if len(cand_idx) == 0:
        return None
    best = min(cand_idx, key=lambda i: abs(idx.t_secs[i] - t_sec))
    return idx.grids[best]


def count_cell_errors(predicted: "np.ndarray", correct: "np.ndarray") -> int:
    """正解ラベルが U (不明) でないセルのみを対象に誤りセル数を数える。"""
    known_mask = correct != COLOR_UNKNOWN
    return int(np.count_nonzero(
        (predicted.astype(np.int64) != correct.astype(np.int64)) & known_mask
    ))


# =============================================================================
# 3. メイン比較処理
# =============================================================================


def compare_one_sample(
    s: LabelSample,
    anchor_idx: "_NpzIndex | None",
    off_idx: "_NpzIndex | None",
    on_idx: "_NpzIndex | None",
) -> CompareResult:
    """アンカー方式で1サンプル分の OFF/ON 誤りセル数を確定する。"""
    anchor_grid = _lookup_anchor_grid(anchor_idx, s.side, s.t_sec)
    if anchor_grid is None or off_idx is None:
        return CompareResult(s, False, None, None, None)

    off_match = _find_bit_exact_match(
        off_idx, s.side, anchor_grid, s.t_sec, ANCHOR_MATCH_WINDOW_SEC,
    )
    if off_match is None:
        return CompareResult(s, False, None, None, None)
    off_grid, verified_t = off_match
    off_errors = count_cell_errors(off_grid, s.correct_grid)

    on_errors: "int | None" = None
    if on_idx is not None:
        on_grid = _find_nearest_snapshot(on_idx, s.side, verified_t, ON_LOOKUP_WINDOW_SEC)
        if on_grid is not None:
            on_errors = count_cell_errors(on_grid, s.correct_grid)
    return CompareResult(s, True, off_errors, on_errors, verified_t)


def compare_all_samples(samples: list[LabelSample]) -> list[CompareResult]:
    """全サンプルについてアンカー突合 + OFF/ON 誤りセル数を計算する。"""
    anchor_cache: dict[str, "_NpzIndex | None"] = {}
    off_cache: dict[str, "_NpzIndex | None"] = {}
    on_cache: dict[str, "_NpzIndex | None"] = {}
    results: list[CompareResult] = []
    for s in samples:
        if s.video_stem not in anchor_cache:
            anchor_cache[s.video_stem] = _load_npz_index(
                ANCHOR_NPZ_DIR / f"{s.video_stem}.npz"
            )
            off_cache[s.video_stem] = _load_npz_index(
                NPZ_ROOT / MODE_OFF / f"{s.video_stem}.npz"
            )
            on_cache[s.video_stem] = _load_npz_index(
                NPZ_ROOT / MODE_ON / f"{s.video_stem}.npz"
            )
        results.append(compare_one_sample(
            s, anchor_cache[s.video_stem], off_cache[s.video_stem], on_cache[s.video_stem],
        ))
    return results


# =============================================================================
# 4. 集計レポート
# =============================================================================


def summarize(results: list[CompareResult]) -> str:
    """OFF/ON 誤りセル数の合計・検収判定・サンプル別内訳をまとめる。"""
    lines: list[str] = []
    n_anchor_missing = sum(1 for r in results if not r.anchor_found)
    off_vals = [r.off_errors for r in results if r.off_errors is not None]
    on_vals = [r.on_errors for r in results if r.on_errors is not None]
    off_total = sum(off_vals)
    on_total = sum(on_vals)

    lines.append(f"サンプル総数: {len(results)} 件")
    lines.append(
        f"アンカー突合失敗 (要調査): {n_anchor_missing} 件 "
        f"(OFF突合成功={len(off_vals)}件 / ON突合成功={len(on_vals)}件)"
    )
    verdict = "合格" if off_total == EXPECTED_OFF_TOTAL_ERRORS else "★不合格★"
    lines.append(
        f"[検収] OFF誤りセル合計={off_total} (期待値={EXPECTED_OFF_TOTAL_ERRORS}) → {verdict}"
    )
    lines.append(f"ON誤りセル合計={on_total}")
    lines.append("")
    lines.append("--- サンプル別内訳 (video/t_sec/side(status): OFF誤り→ON誤り) ---")
    for r in results:
        s = r.sample
        off_s = "N/A(異常)" if r.off_errors is None else str(r.off_errors)
        on_s = "N/A(異常)" if r.on_errors is None else str(r.on_errors)
        anchor_s = "" if r.anchor_found else " [アンカー突合失敗]"
        lines.append(
            f"  video_{s.video_stem} t={s.t_sec:.1f} {s.side} ({s.status}): "
            f"{off_s} → {on_s}{anchor_s}"
        )
    return "\n".join(lines)


def main() -> None:
    samples = load_label_samples()
    print(f"[1/2] ラベル読込: {len(samples)} 件 (video={len({s.video_stem for s in samples})}本)")
    results = compare_all_samples(samples)
    print("[2/2] アンカー突合 + 誤りセル数比較")
    print(summarize(results))


if __name__ == "__main__":
    main()
