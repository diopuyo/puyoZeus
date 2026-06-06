"""
GRAVITY_SETTLE設計パラメータ裏取り測定スクリプト (read-only, 実装変更なし)

測定項目:
  A: chain→stable遷移直後の raw_cnn ぷよ数静止所要frame分布
  B: corruption種別 (色→空 vs 空→色 vs 色→色) の内訳比較
  C: physics_fix_changed_cells の遷移直後集中パターン
  D: raw_cnn ちらつき区間 (エフェクト残光持続フレーム数)

使用方法:
  python scripts/investigate_gravity_settle.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# -------------------------
# 定数
# -------------------------
PROJ = Path(__file__).parent.parent
DATA_DIR = PROJ / "data" / "verify" / "viz"

# 分析対象ファイル
FILES = {
    "v89_caseX":   DATA_DIR / "v89_match01_caseX_2026-06-05.jsonl",
    "v70_caseX":   DATA_DIR / "v70_match02_caseX_2026-06-05.jsonl",
    "v89_glowOFF": DATA_DIR / "v89_match01_glowOFF_2026-06-04.jsonl",  # baseline(過剰保持)
}

# 盤面定数
ROWS = 13
COLS = 6
COLOR_EMPTY = 0
COLOR_UNKNOWN = 10
SETTLE_ANALYSIS_WINDOW = 30   # 遷移後に観察するframe数上限
STABLE_DIFF_THRESHOLD = 2     # ぷよ数差がこれ未満なら「静止」と判定
POST_CHAIN_CORR_WINDOW = 20   # corruption種別を集計する「遷移後フレーム数」

# -------------------------
# ユーティリティ
# -------------------------

def load_jsonl(path: Path) -> list[dict]:
    """JSONLを全行ロードしてdictリストを返す"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def puyo_count(board: list[list[int]]) -> int:
    """13×6 boardのうち非空(0)・非unknown(10)セル数を返す"""
    count = 0
    for row in board:
        for v in row:
            if v != COLOR_EMPTY and v != COLOR_UNKNOWN:
                count += 1
    return count


def board_diff(b1: list[list[int]], b2: list[list[int]]) -> int:
    """2つのboardで値が異なるセル数を返す"""
    diff = 0
    for r in range(ROWS):
        for c in range(COLS):
            if b1[r][c] != b2[r][c]:
                diff += 1
    return diff


def find_chain_to_stable_transitions(frames: list[dict], side: str) -> list[int]:
    """
    p{side}_stateが chain→stable に変わる遷移インデックスのリストを返す。
    返り値: 遷移後の最初のstableフレームのインデックス
    """
    state_key = f"p{side}_state"
    transitions = []
    for i in range(1, len(frames)):
        prev = frames[i-1].get(state_key, "")
        curr = frames[i].get(state_key, "")
        if prev == "chain" and curr == "stable":
            transitions.append(i)
    return transitions


def classify_corruption_cells(
    cnn_board: list[list[int]],
    hsv_board: list[list[int]],
    confirmed: list[list[int]]
) -> dict[str, int]:
    """
    consensus(CNN==HSV, どちらも非UNKNOWN) != confirmed のセルを分類する。
    返り値: {"color_to_empty": N, "empty_to_color": N, "color_to_color": N}
    """
    counts = {"color_to_empty": 0, "empty_to_color": 0, "color_to_color": 0}
    for r in range(ROWS):
        for c in range(COLS):
            cv = cnn_board[r][c]
            hv = hsv_board[r][c]
            cfv = confirmed[r][c]
            # consensus条件: CNN==HSV かつ どちらも非UNKNOWN
            if cv == hv and cv != COLOR_UNKNOWN and hv != COLOR_UNKNOWN:
                if cv != cfv:
                    # corruption発生
                    if cfv == COLOR_EMPTY and cv != COLOR_EMPTY:
                        # confirmed=空, consensus=色 → 色→空(confirmed側が空を採用)
                        counts["color_to_empty"] += 1
                    elif cfv != COLOR_EMPTY and cv == COLOR_EMPTY:
                        # confirmed=色, consensus=空 → 空→色(confirmedに前フレーム色が残留)
                        counts["empty_to_color"] += 1
                    else:
                        counts["color_to_color"] += 1
    return counts


# -------------------------
# 測定A: 静止所要frame分布
# -------------------------

def measure_settle_frames(
    frames: list[dict], side: str, label: str
) -> None:
    """
    chain→stable遷移後、raw_cnn ぷよ数が安定するまでのframe数を測定する。
    「安定」= 連続してABS(今のぷよ数 - 前のぷよ数) < STABLE_DIFF_THRESHOLD が2frame続く
    """
    transitions = find_chain_to_stable_transitions(frames, side)
    cnn_key = f"p{side}_raw_cnn_board"
    settle_frames_list = []

    print(f"\n=== [A] 静止所要frame分布 [{label} side={side}] ===")
    print(f"  chain→stable遷移回数: {len(transitions)}")

    for t_idx in transitions:
        t_sec = frames[t_idx].get("t_sec", "?")
        counts = []
        for i in range(t_idx, min(t_idx + SETTLE_ANALYSIS_WINDOW, len(frames))):
            brd = frames[i].get(cnn_key)
            if brd is None:
                counts.append(None)
            else:
                counts.append(puyo_count(brd))

        # 安定判定: diff < STABLE_DIFF_THRESHOLD が2frame連続する最初のframe
        settle_f = None
        for i in range(1, len(counts)):
            if counts[i] is None or counts[i-1] is None:
                continue
            if abs(counts[i] - counts[i-1]) < STABLE_DIFF_THRESHOLD:
                # もう1frame確認
                if i + 1 < len(counts) and counts[i+1] is not None:
                    if abs(counts[i+1] - counts[i]) < STABLE_DIFF_THRESHOLD:
                        settle_f = i - 1  # 変化が収まり始めたframe(0起算)
                        break
                else:
                    settle_f = i - 1
                    break

        if settle_f is None:
            settle_f = SETTLE_ANALYSIS_WINDOW  # 上限に達した

        settle_frames_list.append(settle_f)
        print(f"  t={t_sec:.2f}s 遷移: settle={settle_f}f  ぷよ数推移={counts[:min(15,len(counts))]}")

    if settle_frames_list:
        import statistics
        print(f"\n  --- 集計 ---")
        print(f"  件数: {len(settle_frames_list)}")
        print(f"  中央値: {statistics.median(settle_frames_list):.1f} frame")
        print(f"  平均:   {statistics.mean(settle_frames_list):.1f} frame")
        print(f"  最大:   {max(settle_frames_list)} frame")
        print(f"  分布: {sorted(settle_frames_list)}")


# -------------------------
# 測定B: corruption種別
# -------------------------

def measure_corruption_types(
    frames: list[dict], side: str, label: str,
    post_window: int = POST_CHAIN_CORR_WINDOW
) -> dict:
    """
    chain→stable遷移後のPOST_CHAIN_CORR_WINDOWフレームと
    それ以外のstableフレームで、corruption種別を集計する。
    """
    transitions_set = set(find_chain_to_stable_transitions(frames, side))
    # 遷移後windowフレームのインデックスセット
    post_chain_indices = set()
    for t in transitions_set:
        for offset in range(post_window):
            if t + offset < len(frames):
                post_chain_indices.add(t + offset)

    cnn_key = f"p{side}_raw_cnn_board"
    hsv_key = f"p{side}_raw_hsv_board"
    conf_key = f"p{side}_confirmed"
    state_key = f"p{side}_state"

    post_totals = defaultdict(int)
    other_totals = defaultdict(int)
    post_stable_frames = 0
    other_stable_frames = 0

    for i, frm in enumerate(frames):
        if frm.get(state_key) != "stable":
            continue
        cnn = frm.get(cnn_key)
        hsv = frm.get(hsv_key)
        conf = frm.get(conf_key)
        if cnn is None or hsv is None or conf is None:
            continue

        result = classify_corruption_cells(cnn, hsv, conf)

        if i in post_chain_indices:
            post_stable_frames += 1
            for k, v in result.items():
                post_totals[k] += v
        else:
            other_stable_frames += 1
            for k, v in result.items():
                other_totals[k] += v

    print(f"\n=== [B] corruption種別 [{label} side={side}] ===")
    print(f"  遷移後{post_window}f以内のstableフレーム数: {post_stable_frames}")
    print(f"  それ以外のstableフレーム数:               {other_stable_frames}")

    def _rate(d: dict, frames_n: int) -> None:
        total_corr = sum(d.values())
        if frames_n == 0:
            print("    (フレームなし)")
            return
        print(f"    corruption総セル数: {total_corr}  (平均{total_corr/frames_n:.2f}/frame)")
        for k, v in d.items():
            pct = 100 * v / total_corr if total_corr > 0 else 0
            print(f"      {k}: {v} ({pct:.1f}%)")

    print(f"  遷移直後({post_window}f以内):")
    _rate(post_totals, post_stable_frames)
    print(f"  通常stable期間:")
    _rate(other_totals, other_stable_frames)

    return {
        "post": dict(post_totals),
        "other": dict(other_totals),
        "post_frames": post_stable_frames,
        "other_frames": other_stable_frames,
    }


# -------------------------
# 測定C: physics_fix集中
# -------------------------

def measure_physics_fix_concentration(
    frames: list[dict], side: str, label: str
) -> None:
    """
    chain→stable遷移後0-10フレームの physics_fix_changed_cells 件数を測定し、
    遷移直後に集中するかを確認する。
    """
    transitions = find_chain_to_stable_transitions(frames, side)
    pf_key = f"p{side}_physics_fix_changed_cells"

    print(f"\n=== [C] physics_fix_changed_cells 集中分析 [{label} side={side}] ===")
    print(f"  遷移回数: {len(transitions)}")

    per_offset = defaultdict(list)  # offset -> セル数リスト

    for t_idx in transitions:
        t_sec = frames[t_idx].get("t_sec", "?")
        row_data = []
        for offset in range(11):  # 0-10
            fi = t_idx + offset
            if fi >= len(frames):
                break
            pf = frames[fi].get(pf_key, [])
            n = len(pf)
            per_offset[offset].append(n)
            row_data.append(n)
        print(f"  t={t_sec:.2f}s: offset0-10 pf件数={row_data}")

    print(f"\n  --- offsetごとの平均physics_fix件数 ---")
    for offset in range(11):
        vals = per_offset[offset]
        if vals:
            avg = sum(vals) / len(vals)
            mx = max(vals)
            print(f"    offset+{offset}f: 平均{avg:.2f}, 最大{mx}, n={len(vals)}")


# -------------------------
# 測定D: raw_cnn ちらつき区間
# -------------------------

def measure_cnn_flicker(
    frames: list[dict], side: str, label: str
) -> None:
    """
    chain→stable遷移直後に raw_cnn_board のセルが変動する区間(frame数)を測定する。
    各セルについて「異なる値が出た最後のframe」を遷移からの距離で測る。
    """
    transitions = find_chain_to_stable_transitions(frames, side)
    cnn_key = f"p{side}_raw_cnn_board"
    FLICKER_WINDOW = 20

    print(f"\n=== [D] raw_cnn ちらつき区間 [{label} side={side}] ===")
    print(f"  遷移回数: {len(transitions)}")

    flicker_durations = []

    for t_idx in transitions:
        t_sec = frames[t_idx].get("t_sec", "?")
        window_frames = []
        for fi in range(t_idx, min(t_idx + FLICKER_WINDOW, len(frames))):
            b = frames[fi].get(cnn_key)
            if b is not None:
                window_frames.append(b)

        if len(window_frames) < 2:
            continue

        # セルごとに最後に変化したoffsetを測定
        max_last_change = 0
        ref = window_frames[0]
        for r in range(ROWS):
            for c in range(COLS):
                last_change = 0
                for fi in range(1, len(window_frames)):
                    if window_frames[fi][r][c] != window_frames[fi-1][r][c]:
                        last_change = fi
                if last_change > max_last_change:
                    max_last_change = last_change

        flicker_durations.append(max_last_change)
        print(f"  t={t_sec:.2f}s: 最後の変化offset={max_last_change}f")

    if flicker_durations:
        import statistics
        print(f"\n  --- 集計 ---")
        print(f"  件数: {len(flicker_durations)}")
        print(f"  中央値: {statistics.median(flicker_durations):.1f} frame")
        print(f"  平均:   {statistics.mean(flicker_durations):.1f} frame")
        print(f"  最大:   {max(flicker_durations)} frame")
        print(f"  分布: {sorted(flicker_durations)}")


# -------------------------
# baseline比較: corruption総量
# -------------------------

def compare_total_corruption(
    frames_casex: list[dict], frames_baseline: list[dict],
    side: str, label_x: str, label_b: str
) -> None:
    """案X vs baseline でstableフレームのcorruption総量を比較する"""

    def _total(frames: list[dict]) -> tuple[int, int, dict]:
        cnn_key = f"p{side}_raw_cnn_board"
        hsv_key = f"p{side}_raw_hsv_board"
        conf_key = f"p{side}_confirmed"
        state_key = f"p{side}_state"
        total = defaultdict(int)
        stable_frames = 0
        for frm in frames:
            if frm.get(state_key) != "stable":
                continue
            cnn = frm.get(cnn_key)
            hsv = frm.get(hsv_key)
            conf = frm.get(conf_key)
            if cnn is None or hsv is None or conf is None:
                continue
            result = classify_corruption_cells(cnn, hsv, conf)
            for k, v in result.items():
                total[k] += v
            stable_frames += 1
        return stable_frames, sum(total.values()), dict(total)

    sf_x, tot_x, detail_x = _total(frames_casex)
    sf_b, tot_b, detail_b = _total(frames_baseline)

    print(f"\n=== [比較] corruption総量: {label_x} vs {label_b} (side={side}) ===")
    print(f"  {label_x}: stableフレーム={sf_x}, 総corruption={tot_x} ({tot_x/sf_x:.2f}/f)")
    print(f"    内訳: {detail_x}")
    print(f"  {label_b}: stableフレーム={sf_b}, 総corruption={tot_b} ({tot_b/sf_b:.2f}/f)")
    print(f"    内訳: {detail_b}")
    if sf_b > 0:
        rate_x = tot_x / sf_x
        rate_b = tot_b / sf_b
        print(f"  差分(caseX - baseline): {rate_x - rate_b:+.3f}/f  ({(rate_x/rate_b - 1)*100:+.1f}%)")


# -------------------------
# メイン
# -------------------------

def main() -> None:
    print("=" * 60)
    print("GRAVITY_SETTLE設計パラメータ裏取り測定")
    print("=" * 60)

    # ファイル存在確認
    for key, path in FILES.items():
        exists = path.exists()
        print(f"  {key}: {'OK' if exists else 'NOT FOUND'} ({path.name})")

    print()

    # データロード
    loaded: dict[str, list[dict]] = {}
    for key, path in FILES.items():
        if path.exists():
            loaded[key] = load_jsonl(path)
            print(f"  {key}: {len(loaded[key])} frames loaded")
        else:
            print(f"  {key}: スキップ (ファイル不在)")

    # ---- 測定A: 静止所要frame ----
    for key in ["v89_caseX", "v70_caseX"]:
        if key in loaded:
            measure_settle_frames(loaded[key], "1", key)
            measure_settle_frames(loaded[key], "2", key)

    # ---- 測定B: corruption種別 ----
    for key in ["v89_caseX", "v70_caseX"]:
        if key in loaded:
            measure_corruption_types(loaded[key], "1", key)
            measure_corruption_types(loaded[key], "2", key)

    # ---- 測定C: physics_fix集中 ----
    for key in ["v89_caseX", "v70_caseX"]:
        if key in loaded:
            measure_physics_fix_concentration(loaded[key], "1", key)
            measure_physics_fix_concentration(loaded[key], "2", key)

    # ---- 測定D: ちらつき区間 ----
    for key in ["v89_caseX", "v70_caseX"]:
        if key in loaded:
            measure_cnn_flicker(loaded[key], "1", key)
            measure_cnn_flicker(loaded[key], "2", key)

    # ---- baseline比較 ----
    if "v89_caseX" in loaded and "v89_glowOFF" in loaded:
        compare_total_corruption(
            loaded["v89_caseX"], loaded["v89_glowOFF"],
            "1", "v89_caseX", "v89_glowOFF(baseline)"
        )
        compare_total_corruption(
            loaded["v89_caseX"], loaded["v89_glowOFF"],
            "2", "v89_caseX", "v89_glowOFF(baseline)"
        )

    print("\n" + "=" * 60)
    print("測定完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
