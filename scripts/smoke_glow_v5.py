"""v5 スモーク検証スクリプト.

v4 と v5 の board_log JSONL を比較して、
cell(2,3) を含む上部セルのおじゃま誤認 frame 数の変化を確認する。

使い方:
    python -m scripts.smoke_glow_v5 \
        --v4-log data/verify/viz/v89_match01_glowV4_2026-06-04.jsonl \
        --v5-log data/verify/viz/v89_match01_glowV5_2026-06-04.jsonl \
        --t-start 66 --t-end 72
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 検証対象の行範囲 (発光で影響を受ける上部)
SMOKE_ROW_MAX: int = 5  # row 0〜4 (上から5行)

# おじゃまカラーコード
COLOR_OJAMA: int = 9

# confirmed_board フィールド候補 (visualize_recognition の --dump-board-log-detailed 形式)
CONFIRMED_KEYS_1P: list[str] = [
    "p1_confirmed", "confirmed_1p", "board_1p", "confirmed_p1", "p1", "1p_confirmed",
]
# raw_cnn / raw_hsv フィールド候補 (detailed jsonl のフィールド名)
RAW_CNN_KEYS_1P: list[str] = ["p1_raw_cnn_board", "p1_raw_cnn", "raw_cnn_1p", "raw_cnn_p1"]
RAW_HSV_KEYS_1P: list[str] = ["p1_raw_hsv_board", "p1_raw_hsv", "raw_hsv_1p", "raw_hsv_p1"]


def _load_jsonl(path: Path) -> list[dict]:
    """JSONL を読み込み dict リストとして返す。"""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _get_t(record: dict) -> float:
    """レコードから時刻 (秒) を取得する。

    visualize_recognition の --dump-board-log-detailed は t_sec フィールドを使用。
    """
    return float(record.get("t_sec", record.get("t", record.get("timestamp", 0))))


def _get_board(record: dict, key_candidates: list[str]) -> list[list[int]] | None:
    """key_candidates の優先順でフィールドを探し、最初に見つかったものを返す。"""
    for key in key_candidates:
        if key in record:
            return record[key]
    return None


def _get_cell(board_grid: list[list[int]], r: int, c: int) -> int:
    """board_grid[r][c] の色を返す。"""
    return int(board_grid[r][c])


def _count_ojama_in_upper(board_grid: list[list[int]]) -> int:
    """board_grid の上部 SMOKE_ROW_MAX 行でおじゃまセル数を数える。"""
    count = 0
    for r in range(SMOKE_ROW_MAX):
        for v in board_grid[r]:
            if int(v) == COLOR_OJAMA:
                count += 1
    return count


def _filter_by_time(records: list[dict], t_start: float, t_end: float) -> list[dict]:
    """t_start〜t_end の範囲のレコードのみ返す。"""
    return [r for r in records if t_start <= _get_t(r) <= t_end]


def _analyze_cell(
    v4_by_t: dict[float, dict],
    v5_by_t: dict[float, dict],
    cell_r: int,
    cell_c: int,
) -> tuple[int, int, int]:
    """cell(cell_r, cell_c) の v4 vs v5 おじゃま誤認 frame 数を比較する。

    Returns:
        (v4_ojama_frames, v5_ojama_frames, restored_frames)
    """
    v4_ojama_frames = 0
    v5_ojama_frames = 0
    restored_frames = 0

    print(f"=== cell({cell_r},{cell_c}) の 1P confirmed: v4 vs v5 ===")
    print(f"{'t':>8} | {'v4_conf':>8} | {'v5_conf':>8} | {'変化'}")
    print("-" * 55)

    for t, rec4 in sorted(v4_by_t.items()):
        rec5 = v5_by_t.get(t)
        board4 = _get_board(rec4, CONFIRMED_KEYS_1P)
        board5 = _get_board(rec5, CONFIRMED_KEYS_1P) if rec5 else None
        v4_v = _get_cell(board4, cell_r, cell_c) if board4 else -1
        v5_v = _get_cell(board5, cell_r, cell_c) if board5 else -1

        if v4_v == COLOR_OJAMA:
            v4_ojama_frames += 1
        if v5_v == COLOR_OJAMA:
            v5_ojama_frames += 1
        if v4_v == COLOR_OJAMA and v5_v != COLOR_OJAMA:
            restored_frames += 1

        change = ""
        if v4_v == COLOR_OJAMA and v5_v != COLOR_OJAMA:
            change = f"O→{v5_v} (復元!)"
        elif v4_v != v5_v:
            change = f"{v4_v}→{v5_v} (変化)"

        if v4_v == COLOR_OJAMA or change:
            print(f"{t:8.2f} | {v4_v:>8} | {v5_v:>8} | {change}")

    return v4_ojama_frames, v5_ojama_frames, restored_frames


def _analyze_upper_total(
    v4_by_t: dict[float, dict],
    v5_by_t: dict[float, dict],
    t_start: float,
    t_end: float,
) -> None:
    """上部 SMOKE_ROW_MAX 行の 1P おじゃまセル合計を v4 vs v5 で比較する。"""
    v4_total = 0
    v5_total = 0
    for t, rec4 in v4_by_t.items():
        rec5 = v5_by_t.get(t)
        board4 = _get_board(rec4, CONFIRMED_KEYS_1P)
        board5 = _get_board(rec5, CONFIRMED_KEYS_1P) if rec5 else None
        if board4:
            v4_total += _count_ojama_in_upper(board4)
        if board5:
            v5_total += _count_ojama_in_upper(board5)

    print(f"=== 上部 row 0〜{SMOKE_ROW_MAX - 1} の 1P おじゃまセル合計 (t={t_start}〜{t_end}s) ===")
    print(f"  v4 合計: {v4_total} セル×frame")
    print(f"  v5 合計: {v5_total} セル×frame")
    delta = v4_total - v5_total
    print(f"  削減: {delta} ({delta / max(v4_total, 1) * 100:.1f}% 減)")


def _check_false_restore(
    v4_by_t: dict[float, dict],
    v5_by_t: dict[float, dict],
) -> int:
    """真おじゃま(raw_cnn=O かつ raw_hsv=O)を v5 が誤復元した事例を数える。

    v5 で v4 の O が非 O になったセルのうち、
    v5 の raw_cnn と raw_hsv が両方おじゃまを示すケースを検知する。

    Returns:
        false_restore_count: 誤復元件数 (0 なら OK)。
    """
    print("=== 真おじゃま不触チェック (raw_cnn=O かつ raw_hsv=O なのに v5 が復元した事例) ===")
    false_restore_count = 0
    for t, rec5 in v5_by_t.items():
        rec4 = v4_by_t.get(t)
        if rec4 is None:
            continue
        board4 = _get_board(rec4, CONFIRMED_KEYS_1P)
        board5 = _get_board(rec5, CONFIRMED_KEYS_1P)
        raw_cnn5 = _get_board(rec5, RAW_CNN_KEYS_1P)
        raw_hsv5 = _get_board(rec5, RAW_HSV_KEYS_1P)
        if board4 is None or board5 is None:
            continue

        for r in range(SMOKE_ROW_MAX):
            for c in range(6):
                v4_v = _get_cell(board4, r, c)
                v5_v = _get_cell(board5, r, c)
                # v4=O かつ v5=非O → 復元が起きた
                if v4_v != COLOR_OJAMA or v5_v == COLOR_OJAMA:
                    continue
                # raw_cnn/raw_hsv が両方おじゃま → 真おじゃまを誤復元
                if raw_cnn5 and raw_hsv5:
                    cnn_v = _get_cell(raw_cnn5, r, c)
                    hsv_v = _get_cell(raw_hsv5, r, c)
                    if cnn_v == COLOR_OJAMA and hsv_v == COLOR_OJAMA:
                        print(f"  [警告] t={t:.2f} cell({r},{c}): "
                              f"raw_cnn=O, raw_hsv=O なのに v5 が {v5_v} に復元 (真おじゃま誤復元)")
                        false_restore_count += 1

    if false_restore_count == 0:
        print("  [OK] 真おじゃま誤復元なし (raw_cnn=O かつ raw_hsv=O のセルは全て保持)")
    else:
        print(f"  [NG] 真おじゃま誤復元: {false_restore_count} 件")
    return false_restore_count


def analyze(
    v4_log: Path,
    v5_log: Path,
    t_start: float,
    t_end: float,
) -> None:
    """v4 と v5 の JSONL を比較して差分を表示する。"""
    v4_records = _load_jsonl(v4_log)
    v5_records = _load_jsonl(v5_log)

    print(f"[smoke_glow_v5] v4 log: {v4_log} ({len(v4_records)} frames)")
    print(f"[smoke_glow_v5] v5 log: {v5_log} ({len(v5_records)} frames)")
    print(f"[smoke_glow_v5] 対象時刻範囲: t={t_start}〜{t_end}s")
    print()

    v4_filtered = _filter_by_time(v4_records, t_start, t_end)
    v5_filtered = _filter_by_time(v5_records, t_start, t_end)
    print(f"  フィルタ後: v4={len(v4_filtered)} frames, v5={len(v5_filtered)} frames")
    print()

    if v4_filtered:
        print("[smoke_glow_v5] v4 レコードのフィールド一覧 (先頭 25 個):")
        for k in list(v4_filtered[0].keys())[:25]:
            print(f"  {k}")
        print()

    v4_by_t: dict[float, dict] = {_get_t(r): r for r in v4_filtered}
    v5_by_t: dict[float, dict] = {_get_t(r): r for r in v5_filtered}

    # cell(2,3) の比較 (v89 t≈71 で発生した残差の中心)
    v4_ojama, v5_ojama, restored = _analyze_cell(v4_by_t, v5_by_t, cell_r=2, cell_c=3)
    print()
    print(f"=== cell(2,3) O誤認 frame 数 ===")
    print(f"  v4: {v4_ojama} frames がおじゃま誤認")
    print(f"  v5: {v5_ojama} frames がおじゃま誤認")
    print(f"  v5 で復元された frame 数: {restored}")
    print()

    _analyze_upper_total(v4_by_t, v5_by_t, t_start, t_end)
    print()

    _check_false_restore(v4_by_t, v5_by_t)
    print()
    print("[smoke_glow_v5] 完了")


def main() -> int:
    """スモーク検証のエントリポイント。"""
    parser = argparse.ArgumentParser(description="v4 vs v5 glow guard スモーク比較")
    parser.add_argument("--v4-log", type=Path, required=True, help="v4 board_log JSONL パス")
    parser.add_argument("--v5-log", type=Path, required=True, help="v5 board_log JSONL パス")
    parser.add_argument("--t-start", type=float, default=66.0, help="対象開始時刻 (秒)")
    parser.add_argument("--t-end", type=float, default=72.0, help="対象終了時刻 (秒)")
    args = parser.parse_args()

    if not args.v4_log.exists():
        print(f"[ERROR] v4 log が見つかりません: {args.v4_log}")
        return 1
    if not args.v5_log.exists():
        print(f"[ERROR] v5 log が見つかりません: {args.v5_log}")
        return 1

    analyze(args.v4_log, args.v5_log, args.t_start, args.t_end)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
