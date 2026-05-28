"""Q1 全力施策 - 認識不能 frame 抽出スクリプト。

詳細 board log JSONL (visualize_recognition.py --dump-board-log-detailed 出力) を
読み込み、以下の条件を満たす「認識不能」cell/frame を抽出してレポートする:

  抽出条件:
    - confirmed_board の特定 cell が CONTINUOUS_EMPTY_FRAMES フレーム以上
      連続して EMPTY (= 0) のまま
    - かつ、同 frame の raw_cnn_board または raw_hsv_board のいずれかが
      非 EMPTY を返している (= 「ぷよがあるのに認識システムが消した」ケース)

  真因仮説の判定:
    - bg_fp_distance < BG_FP_HYPOTHESIS_THRESHOLD → bg_fp 汚染仮説
    - state が STABLE_WARMUP_FRAMES 以内の STABLE 遷移直後 → warmup 不足仮説
    - raw_cnn で OJAMA (9) が多い → CNN ojama 過判定仮説

Usage:
    PYTHONPATH=. .\\venv\\Scripts\\python -m scripts.extract_unrecognized_frames \\
        --log logs/board_v40_match01_detailed.jsonl \\
        --output logs/unrecognized_v40_match01.json

    # 両動画を一括処理する場合:
    PYTHONPATH=. .\\venv\\Scripts\\python -m scripts.extract_unrecognized_frames \\
        --log logs/board_v40_detailed.jsonl logs/board_v89_detailed.jsonl \\
        --output logs/unrecognized_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ============================
# 判定定数
# ============================

# confirmed が EMPTY 継続フレーム数の閾値 (= 30 frame ≒ 1 秒 @ 30fps)
CONTINUOUS_EMPTY_FRAMES: int = 30

# bg_fp 汚染仮説の閾値: 距離がこれ未満なら tier1 early EMPTY で消された可能性大
BG_FP_HYPOTHESIS_THRESHOLD: float = 25.0

# warmup 仮説: STABLE 遷移後 N frame 以内を「warmup 期間」と定義
STABLE_WARMUP_FRAMES: int = 12

# COLOR コード定数 (board.py と同値)
COLOR_EMPTY: int = 0
COLOR_RED: int = 1
COLOR_BLUE: int = 2
COLOR_GREEN: int = 3
COLOR_YELLOW: int = 4
COLOR_PURPLE: int = 5
COLOR_OJAMA: int = 9
COLOR_UNKNOWN: int = 10

# ぷよ色セット (= 非 EMPTY / 非 UNKNOWN の色)
PUYO_COLORS: frozenset[int] = frozenset({
    COLOR_RED, COLOR_BLUE, COLOR_GREEN, COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
})

COLOR_NAMES: dict[int, str] = {
    COLOR_EMPTY: "EMPTY",
    COLOR_RED: "RED",
    COLOR_BLUE: "BLUE",
    COLOR_GREEN: "GREEN",
    COLOR_YELLOW: "YELLOW",
    COLOR_PURPLE: "PURPLE",
    COLOR_OJAMA: "OJAMA",
    COLOR_UNKNOWN: "UNKNOWN",
}

# board サイズ定数
BOARD_ROWS: int = 13
BOARD_COLS: int = 6
HIDDEN_ROWS: int = 1  # row 0 は隠し段


# ============================
# ユーティリティ関数
# ============================

def _grid_cell(grid: list[list[int]] | None, row: int, col: int) -> int:
    """grid から cell 値を安全に取得する。grid が None の場合 COLOR_EMPTY を返す。

    Args:
        grid: 13×6 の int リスト。
        row: 行インデックス。
        col: 列インデックス。

    Returns:
        cell 値 (COLOR_* 定数)。
    """
    if grid is None:
        return COLOR_EMPTY
    if row < 0 or row >= len(grid):
        return COLOR_EMPTY
    row_data = grid[row]
    if col < 0 or col >= len(row_data):
        return COLOR_EMPTY
    return int(row_data[col])


def _dist_cell(
    dist_grid: list[list[float]] | None,
    row: int, col: int,
) -> float | None:
    """bg_fp_distance_grid から距離値を安全に取得する。

    Args:
        dist_grid: 13×6 の float リスト (None or -1.0 は「未採取」を意味する)。
        row: 行インデックス。
        col: 列インデックス。

    Returns:
        float 距離値、または None (未採取)。
    """
    if dist_grid is None:
        return None
    if row < 0 or row >= len(dist_grid):
        return None
    row_data = dist_grid[row]
    if col < 0 or col >= len(row_data):
        return None
    v = float(row_data[col])
    return v if v >= 0.0 else None


# ============================
# 主処理関数
# ============================

def _classify_hypothesis(
    bg_dist: float | None,
    state_str: str,
    frames_since_stable: int | None,
    raw_cnn_color: int,
    raw_hsv_color: int,
    bg_fp_threshold: float = BG_FP_HYPOTHESIS_THRESHOLD,
) -> list[str]:
    """認識不能の真因仮説を分類する。

    複数仮説が同時に成立する場合は全て返す。

    Args:
        bg_dist: bg_fp 距離値 (None は bg_fp 未採取)。
        state_str: 現 frame の state 文字列 ("stable" / "tsumo_fall" 等)。
        frames_since_stable: この STABLE 開始からの経過フレーム数 (None = STABLE 外)。
        raw_cnn_color: raw CNN の cell 出力色コード。
        raw_hsv_color: raw HSV の cell 出力色コード。
        bg_fp_threshold: bg_fp 汚染仮説の距離閾値。

    Returns:
        仮説名のリスト (空リスト = 該当なし = "不明")。
    """
    hypotheses: list[str] = []

    # 仮説 1: bg_fp 汚染 (tier1 early EMPTY)
    if bg_dist is not None and bg_dist < bg_fp_threshold:
        hypotheses.append("bg_fp_tier1_early_empty")

    # 仮説 2: warmup 不足 (STABLE 遷移直後)
    if (
        state_str == "stable"
        and frames_since_stable is not None
        and frames_since_stable <= STABLE_WARMUP_FRAMES
    ):
        hypotheses.append("warmup_insufficient")

    # 仮説 3: CNN ojama 過判定 (CNN が OJAMA を返して bg_fp で消された)
    if raw_cnn_color == COLOR_OJAMA and bg_dist is not None and bg_dist < bg_fp_threshold:
        hypotheses.append("cnn_ojama_over_prediction_with_bg_fp")
    elif raw_cnn_color == COLOR_OJAMA:
        hypotheses.append("cnn_ojama_over_prediction")

    # 仮説 4: HSV-only 未検出 (HSV も EMPTY だが CNN は非 EMPTY)
    if raw_hsv_color in PUYO_COLORS and raw_cnn_color not in PUYO_COLORS:
        hypotheses.append("hsv_detected_but_cnn_missed")
    elif raw_cnn_color in PUYO_COLORS and raw_hsv_color not in PUYO_COLORS:
        hypotheses.append("cnn_detected_but_hsv_missed")

    # 仮説 5: bg_fp 未採取 (pre_capture_mode による EMPTY 強制)
    if bg_dist is None:
        hypotheses.append("bg_fp_not_captured_yet")

    return hypotheses if hypotheses else ["unknown"]


def _detect_unrecognized_frames(
    entries: list[dict[str, Any]],
    continuous_empty_frames: int = CONTINUOUS_EMPTY_FRAMES,
    bg_fp_threshold: float = BG_FP_HYPOTHESIS_THRESHOLD,
) -> dict[str, Any]:
    """JSONL エントリ群から認識不能 frame を検出する。

    処理:
      1. 各 cell の confirmed 履歴を追跡し continuous_empty_frames 以上の
         EMPTY 継続を検出
      2. 同区間で raw_cnn or raw_hsv が非 EMPTY を返していれば「認識不能」と判定
      3. 仮説分類 + 統計集計

    Args:
        entries: JSONL の各行を parse した dict のリスト。
        continuous_empty_frames: 認識不能判定の連続 EMPTY フレーム閾値。
        bg_fp_threshold: bg_fp 汚染仮説の距離閾値。

    Returns:
        認識不能 frame レポート dict。
    """
    # cell ごとの confirmed EMPTY 連続状態を追跡
    # key: (side, row, col) → value: {
    #   "empty_streak": int (連続 EMPTY カウント),
    #   "streak_start_frame": int | None,
    #   "streak_cnn_nonempties": list[dict],  # streak 内で非 EMPTY を返した frame
    #   "streak_hsv_nonempties": list[dict],
    # }
    cell_state: dict[tuple[str, int, int], dict] = {}
    for side in ("1P", "2P"):
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                cell_state[(side, r, c)] = {
                    "empty_streak": 0,
                    "streak_start_frame": None,
                    "streak_events": [],
                }

    # STABLE 開始 frame の追跡 (warmup 仮説用)
    # key: side → (last_stable_start_frame, consecutive_stable_count)
    stable_tracker: dict[str, tuple[int, int]] = {
        "1P": (-1, 0),
        "2P": (-1, 0),
    }
    prev_states: dict[str, str] = {"1P": "menu", "2P": "menu"}

    # 集計バッファ
    unrecognized_events: list[dict] = []
    hypothesis_counts: dict[str, int] = defaultdict(int)
    hypothesis_frame_sets: dict[str, set[int]] = defaultdict(set)

    for entry in entries:
        fi = int(entry.get("frame_idx", -1))
        t_sec = float(entry.get("t_sec", 0.0))
        pre_capture = bool(entry.get("pre_capture_mode", False))

        for side in ("1P", "2P"):
            prefix = side.lower().replace("p", "p")  # "1P" → "1p"
            # JSONL key は "p1_" / "p2_" 形式
            key_pfx = "p1_" if side == "1P" else "p2_"
            state_str = str(entry.get(f"{key_pfx}state", "menu"))

            # STABLE 遷移検出
            prev_s = prev_states[side]
            if state_str == "stable" and prev_s != "stable":
                stable_tracker[side] = (fi, 0)
            stable_start, _ = stable_tracker[side]
            frames_since_stable = (
                fi - stable_start if state_str == "stable" and stable_start >= 0 else None
            )
            if state_str == "stable":
                stable_tracker[side] = (
                    stable_start,
                    stable_tracker[side][1] + 1,
                )
            prev_states[side] = state_str

            confirmed_grid = entry.get(f"{key_pfx}confirmed")
            raw_cnn_grid = entry.get(f"{key_pfx}raw_cnn_board")
            raw_hsv_grid = entry.get(f"{key_pfx}raw_hsv_board")
            bg_dist_grid = entry.get(f"{key_pfx}bg_fp_distance_grid")

            for r in range(HIDDEN_ROWS, BOARD_ROWS):
                for c in range(BOARD_COLS):
                    confirmed_v = _grid_cell(confirmed_grid, r, c)
                    cnn_v = _grid_cell(raw_cnn_grid, r, c)
                    hsv_v = _grid_cell(raw_hsv_grid, r, c)
                    bg_dist = _dist_cell(bg_dist_grid, r, c)

                    cs = cell_state[(side, r, c)]

                    if confirmed_v == COLOR_EMPTY:
                        # confirmed が EMPTY → streak カウント
                        cs["empty_streak"] += 1
                        if cs["streak_start_frame"] is None:
                            cs["streak_start_frame"] = fi

                        # raw_cnn or raw_hsv が非 EMPTY → 認識不能 candidate
                        cnn_nonempty = cnn_v in PUYO_COLORS
                        hsv_nonempty = hsv_v in PUYO_COLORS
                        if cnn_nonempty or hsv_nonempty:
                            hyps = _classify_hypothesis(
                                bg_dist, state_str, frames_since_stable,
                                cnn_v, hsv_v,
                                bg_fp_threshold=bg_fp_threshold,
                            )
                            cs["streak_events"].append({
                                "frame_idx": fi,
                                "t_sec": round(t_sec, 3),
                                "state": state_str,
                                "frames_since_stable": frames_since_stable,
                                "confirmed": confirmed_v,
                                "raw_cnn": cnn_v,
                                "raw_hsv": hsv_v,
                                "bg_fp_distance": (
                                    round(bg_dist, 2) if bg_dist is not None else None
                                ),
                                "pre_capture_mode": pre_capture,
                                "hypotheses": hyps,
                            })

                        # 閾値超え → 認識不能イベントとして確定
                        if cs["empty_streak"] >= continuous_empty_frames:
                            if cs["streak_events"]:
                                # 初回閾値超えのみイベント発火 (以降は重複しない)
                                if cs["empty_streak"] == continuous_empty_frames:
                                    all_hyps: list[str] = []
                                    for ev in cs["streak_events"]:
                                        all_hyps.extend(ev["hypotheses"])
                                    # 最頻仮説を primary に
                                    hyp_counter: dict[str, int] = defaultdict(int)
                                    for h in all_hyps:
                                        hyp_counter[h] += 1
                                    primary_hyp = max(
                                        hyp_counter, key=lambda k: hyp_counter[k],
                                    ) if hyp_counter else "unknown"

                                    event: dict = {
                                        "side": side,
                                        "row": r,
                                        "col": c,
                                        "streak_start_frame": cs["streak_start_frame"],
                                        "streak_start_t_sec": round(
                                            (cs["streak_start_frame"] or 0) / max(
                                                1, int(
                                                    cs["streak_events"][0].get(
                                                        "t_sec", 0
                                                    ) / max(
                                                        0.001,
                                                        cs["streak_events"][0].get(
                                                            "frame_idx", 1
                                                        ) / max(1, cs["streak_start_frame"] or 1),
                                                    )
                                                )
                                            ), 3,
                                        ),
                                        "streak_length_at_detection": cs["empty_streak"],
                                        "nonempty_events_in_streak": len(cs["streak_events"]),
                                        "primary_hypothesis": primary_hyp,
                                        "hypothesis_counts": dict(hyp_counter),
                                        "sample_events": cs["streak_events"][:5],
                                    }
                                    unrecognized_events.append(event)
                                    for h in hyp_counter:
                                        hypothesis_counts[h] += 1
                                        hypothesis_frame_sets[h].add(
                                            cs["streak_start_frame"] or fi,
                                        )
                    else:
                        # confirmed が非 EMPTY → streak リセット
                        cs["empty_streak"] = 0
                        cs["streak_start_frame"] = None
                        cs["streak_events"] = []

    # 集計結果
    total_events = len(unrecognized_events)
    hypothesis_summary: dict[str, dict] = {}
    for h, cnt in hypothesis_counts.items():
        hypothesis_summary[h] = {
            "cell_count": cnt,
            "frame_count": len(hypothesis_frame_sets[h]),
        }

    return {
        "total_unrecognized_cell_events": total_events,
        "hypothesis_summary": hypothesis_summary,
        "events": unrecognized_events,
    }


def _generate_text_report(report: dict[str, Any], log_path: str) -> str:
    """テキスト形式のレポートを生成する。

    Args:
        report: _detect_unrecognized_frames() の戻り値。
        log_path: 元 JSONL ファイルのパス (表示用)。

    Returns:
        レポート文字列。
    """
    lines: list[str] = []
    lines.append(f"=== 認識不能 frame レポート ===")
    lines.append(f"入力: {log_path}")
    lines.append(f"認識不能 cell イベント総数: {report['total_unrecognized_cell_events']}")
    lines.append("")
    lines.append("--- 真因仮説サマリー ---")
    for h, info in sorted(
        report["hypothesis_summary"].items(),
        key=lambda kv: -kv[1]["cell_count"],
    ):
        lines.append(
            f"  {h}: cell_count={info['cell_count']} "
            f"frame_count={info['frame_count']}"
        )
    lines.append("")
    lines.append("--- 上位 20 件の認識不能イベント ---")
    for i, ev in enumerate(report["events"][:20]):
        lines.append(
            f"  [{i+1}] side={ev['side']} row={ev['row']} col={ev['col']} "
            f"streak_start_frame={ev['streak_start_frame']} "
            f"streak_len={ev['streak_length_at_detection']} "
            f"nonempty_obs={ev['nonempty_events_in_streak']} "
            f"primary={ev['primary_hypothesis']}"
        )
        for sev in ev.get("sample_events", [])[:2]:
            lines.append(
                f"    frame={sev['frame_idx']} t={sev['t_sec']}s "
                f"state={sev['state']} "
                f"raw_cnn={COLOR_NAMES.get(sev['raw_cnn'], sev['raw_cnn'])} "
                f"raw_hsv={COLOR_NAMES.get(sev['raw_hsv'], sev['raw_hsv'])} "
                f"bg_dist={sev['bg_fp_distance']} "
                f"hyp={sev['hypotheses']}"
            )
    return "\n".join(lines)


# ============================
# エントリポイント
# ============================

def main() -> int:
    """メイン処理。"""
    parser = argparse.ArgumentParser(
        description=(
            "Q1 全力施策: 詳細 board log から認識不能 frame を抽出してレポートする。\n"
            "入力: visualize_recognition.py --dump-board-log-detailed の出力 JSONL。"
        ),
    )
    parser.add_argument(
        "--log", type=Path, nargs="+", required=True,
        help="詳細 board log JSONL ファイル (複数指定可)。",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="レポート出力 JSON ファイル。省略時は stdout にテキスト出力。",
    )
    parser.add_argument(
        "--continuous-empty-frames", type=int, default=CONTINUOUS_EMPTY_FRAMES,
        help=f"認識不能判定の連続 EMPTY フレーム閾値 (デフォルト: {CONTINUOUS_EMPTY_FRAMES})。",
    )
    parser.add_argument(
        "--bg-fp-threshold", type=float, default=BG_FP_HYPOTHESIS_THRESHOLD,
        help=f"bg_fp 汚染仮説の距離閾値 (デフォルト: {BG_FP_HYPOTHESIS_THRESHOLD})。",
    )
    args = parser.parse_args()

    # 引数で閾値を上書き (グローバル変数を回避して関数引数で渡す)
    eff_empty_frames: int = args.continuous_empty_frames
    eff_bg_threshold: float = args.bg_fp_threshold

    combined_report: dict[str, Any] = {
        "per_log": {},
        "combined_hypothesis_summary": defaultdict(lambda: {"cell_count": 0, "frame_count": 0}),
        "combined_total_events": 0,
    }

    for log_path in args.log:
        if not log_path.exists():
            print(f"[ERROR] log ファイルが見つかりません: {log_path}", file=sys.stderr)
            continue

        print(f"[extract] 処理中: {log_path}", file=sys.stderr)
        entries: list[dict] = []
        with log_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        print(f"[extract]   {len(entries)} frame のエントリを読み込み", file=sys.stderr)
        report = _detect_unrecognized_frames(
            entries,
            continuous_empty_frames=eff_empty_frames,
            bg_fp_threshold=eff_bg_threshold,
        )
        combined_report["per_log"][str(log_path)] = report
        combined_report["combined_total_events"] += report["total_unrecognized_cell_events"]
        for h, info in report["hypothesis_summary"].items():
            combined_report["combined_hypothesis_summary"][h]["cell_count"] += info["cell_count"]
            combined_report["combined_hypothesis_summary"][h]["frame_count"] += info["frame_count"]

        # テキストレポートを stderr に出力
        text = _generate_text_report(report, str(log_path))
        print(text, file=sys.stderr)
        print("", file=sys.stderr)

    # combined サマリー出力
    combined_report["combined_hypothesis_summary"] = dict(
        combined_report["combined_hypothesis_summary"]
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fp:
            json.dump(combined_report, fp, ensure_ascii=False, indent=2)
        print(f"[extract] レポートを保存: {args.output}", file=sys.stderr)
    else:
        print(json.dumps(combined_report, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
