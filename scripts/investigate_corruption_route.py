"""corruption 侵入経路実証スクリプト (2026-06-03).

既存 board_log (JSONL) を解析し、STABLE フレームの corruption セルを
侵入経路別に分類する。新規の重い動画ランは行わず、既存 board_log のみ使用。

使い方:
    python scripts/investigate_corruption_route.py

出力:
    - corruption 種別分布 (color→color / color→empty / empty→color 等)
    - 侵入経路別件数 (a: physics_fix/infer_placement由来, b: T2フリーズ由来,
                      c: constraint_fill由来, d: その他)
    - infer_placement 誤色が確認できるケースの詳細サンプル
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# ============================================================
# 定数
# ============================================================

# 解析対象 board_log ファイル (相対パス基点 = puyo_analyzer root)
BOARD_LOG_FILES: list[str] = [
    "data/verify/viz/v89_match01_D_2026-06-03.jsonl",
    "data/verify/viz/v89_match02_D_2026-06-03.jsonl",
    "data/verify/viz/v70_match02_formulaD_2026-06-02.jsonl",
]

# 色コード → 名前
COLOR_NAMES: dict[int, str] = {
    0: "EMPTY",
    1: "RED",
    2: "BLUE",
    3: "GREEN",
    4: "YELLOW",
    5: "PURPLE",
    9: "OJAMA",
    10: "UNKNOWN",
}

# T2フリーズ判定: confirmed が連続 N フレーム以上同じ値のままで
# consensus がそれと異なり続ければ「T2 フリーズ由来」と判定する
T2_FREEZE_MIN_FRAMES: int = 3

# physics_fix 由来判定: corruption 発見時点から過去 N フレーム以内に
# 当該セルで physics_fix_changed_cells が発火していれば「infer 由来」
PHYSICS_FIX_LOOKBACK_FRAMES: int = 30

# 解析サンプル出力: 各侵入経路の代表例を最大何件表示するか
MAX_SAMPLE_OUTPUT: int = 5

# 安定状態名 (board_log の state フィールド値)
STABLE_STATE: str = "stable"

# 有色コード集合 (EMPTY/UNKNOWN/OJAMA 以外)
COLORED_CODES: frozenset[int] = frozenset({1, 2, 3, 4, 5})


def is_colored(v: int) -> bool:
    """有色ぷよか (1-5)."""
    return v in COLORED_CODES


def color_name(v: int) -> str:
    """色コード → 表示名."""
    return COLOR_NAMES.get(v, f"?{v}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL ファイルを全行読み込む."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_board(frame: dict[str, Any], side: str, key: str) -> list[list[int]] | None:
    """frame から指定 side/key の盤面を取得する.

    key 例: "confirmed", "raw_cnn_board", "raw_hsv_board"
    side: "p1" or "p2"
    """
    full_key = f"{side}_{key}"
    val = frame.get(full_key)
    if val is None:
        return None
    return val  # list[list[int]], shape [13][6]


def board_get(board: list[list[int]], r: int, c: int) -> int:
    """盤面配列から (r, c) の値を取得."""
    return int(board[r][c])


def analyze_corruption_frame(
    frame: dict[str, Any],
    side: str,
) -> list[dict[str, Any]]:
    """1フレームの corruption セルを全列挙する.

    corruption 定義:
        STABLE 状態 かつ
        raw_cnn == raw_hsv (consensus, 両方非 UNKNOWN) かつ
        confirmed != consensus

    Returns:
        各 corruption セルの dict リスト。
        {row, col, confirmed_v, consensus_v, cnn_v, hsv_v,
         kind: "color→color" | "color→empty" | "empty→color" | "other",
         physics_fix_cells, constraint_fill_cells}
    """
    state = frame.get(f"{side}_state", "")
    if state != STABLE_STATE:
        return []

    confirmed = get_board(frame, side, "confirmed")
    raw_cnn = get_board(frame, side, "raw_cnn_board")
    raw_hsv = get_board(frame, side, "raw_hsv_board")
    if confirmed is None or raw_cnn is None or raw_hsv is None:
        return []

    # physics_fix / constraint_fill の変更セル集合
    pfc: list[list[int]] = frame.get(f"{side}_physics_fix_changed_cells") or []
    cfc: list[list[int]] = frame.get(f"{side}_constraint_fill_changed_cells") or []

    # [row, col, old_val, new_val] → (row, col): new_val のマップ
    pfc_map: dict[tuple[int, int], int] = {
        (int(x[0]), int(x[1])): int(x[3]) for x in pfc
    }
    cfc_map: dict[tuple[int, int], int] = {
        (int(x[0]), int(x[1])): int(x[3]) for x in cfc
    }

    corruptions = []
    for r in range(13):
        for c in range(6):
            cnn_v = board_get(raw_cnn, r, c)
            hsv_v = board_get(raw_hsv, r, c)
            # consensus 条件: cnn == hsv かつ両方非 UNKNOWN (10)
            if cnn_v == 10 or hsv_v == 10:
                continue
            if cnn_v != hsv_v:
                continue
            consensus_v = cnn_v
            conf_v = board_get(confirmed, r, c)
            if conf_v == consensus_v:
                continue  # 一致 = corruption なし

            # corruption 種別分類
            if is_colored(conf_v) and consensus_v == 0:
                kind = "color→empty"
            elif conf_v == 0 and is_colored(consensus_v):
                kind = "empty→color"
            elif is_colored(conf_v) and is_colored(consensus_v) and conf_v != consensus_v:
                kind = "color→color"
            else:
                kind = "other"

            corruptions.append({
                "row": r,
                "col": c,
                "confirmed_v": conf_v,
                "consensus_v": consensus_v,
                "cnn_v": cnn_v,
                "hsv_v": hsv_v,
                "kind": kind,
                "pfc_map": pfc_map,   # この frame の physics_fix 変更
                "cfc_map": cfc_map,   # この frame の constraint_fill 変更
            })
    return corruptions


def classify_route(
    cell_key: tuple[int, int],
    conf_v: int,
    frame_idx: int,
    history: dict[tuple[int, int], list[tuple[int, int, int]]],
    pfc_history: dict[tuple[int, int], list[tuple[int, int, int]]],
) -> str:
    """corruption セルの侵入経路を分類する.

    Args:
        cell_key: (row, col)
        conf_v: corruption として検出された confirmed 値
        frame_idx: 現在の frame インデックス
        history: {(row,col): [(frame_idx, confirmed_v, consensus_v), ...]}
                  過去のフレーム履歴 (直近 N フレーム)
        pfc_history: {(row,col): [(frame_idx, old_v, new_v), ...]}
                     過去の physics_fix 書込み履歴

    Returns:
        "a_infer_placement": infer_placement/physics_fix 起源
        "b_t2_freeze": T2 フリーズ起源
        "c_constraint_fill": constraint_fill 起源 (現在 OFF のはず)
        "d_other": 不明
    """
    r, c = cell_key

    # 経路 a: physics_fix 書込みが過去 PHYSICS_FIX_LOOKBACK_FRAMES 以内に発火し、
    # その new_val が conf_v と一致するか
    if cell_key in pfc_history:
        for (pf_frame, old_v, new_v) in pfc_history[cell_key]:
            if (
                frame_idx - pf_frame <= PHYSICS_FIX_LOOKBACK_FRAMES
                and new_v == conf_v
            ):
                return "a_infer_placement"

    # 経路 b: confirmed が T2_FREEZE_MIN_FRAMES 以上連続して conf_v のまま変化せず、
    # 一方 consensus がそれと異なる値を継続している
    if cell_key in history:
        hist = history[cell_key]
        if len(hist) >= T2_FREEZE_MIN_FRAMES:
            # 末尾 T2_FREEZE_MIN_FRAMES フレームを確認
            tail = hist[-T2_FREEZE_MIN_FRAMES:]
            # 全フレームで confirmed == conf_v かつ consensus != conf_v なら凍結判定
            all_frozen = all(
                h_conf == conf_v and h_cons != conf_v
                for (_, h_conf, h_cons) in tail
            )
            if all_frozen:
                return "b_t2_freeze"

    return "d_other"


def analyze_file(path: Path) -> dict[str, Any]:
    """1ファイルを解析し、corruption 統計と経路分類を返す."""
    print(f"\n{'='*60}")
    print(f"解析: {path.name}")
    print(f"{'='*60}")

    frames = load_jsonl(path)
    print(f"  総フレーム数: {len(frames)}")

    sides = ["p1", "p2"]

    # 統計集計用
    kind_count: dict[str, int] = defaultdict(int)
    route_count: dict[str, int] = defaultdict(int)
    total_stable_frames = 0
    total_stable_cells = 0

    # サンプル収集用 (経路別)
    route_samples: dict[str, list[dict]] = defaultdict(list)

    # 状態追跡用 (cell ごとの confirmed/consensus 履歴)
    # key: (side, row, col) → [(frame_idx, confirmed_v, consensus_v), ...]
    cell_history: dict[tuple[str, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    # physics_fix 書込み履歴: (side, row, col) → [(frame_idx, old_v, new_v), ...]
    pfc_hist: dict[tuple[str, int, int], list[tuple[int, int, int]]] = defaultdict(list)

    for frame in frames:
        frame_idx = int(frame.get("frame_idx", 0))

        for side in sides:
            state = frame.get(f"{side}_state", "")

            # physics_fix 履歴を更新 (state に関わらず記録)
            pfc: list[list[int]] = frame.get(f"{side}_physics_fix_changed_cells") or []
            for entry in pfc:
                r, c_col, old_v, new_v = int(entry[0]), int(entry[1]), int(entry[2]), int(entry[3])
                pfc_hist[(side, r, c_col)].append((frame_idx, old_v, new_v))
                # 履歴は直近 100 フレームのみ保持 (メモリ節約)
                if len(pfc_hist[(side, r, c_col)]) > 100:
                    pfc_hist[(side, r, c_col)] = pfc_hist[(side, r, c_col)][-100:]

            if state != STABLE_STATE:
                continue

            total_stable_frames += 1

            confirmed = get_board(frame, side, "confirmed")
            raw_cnn = get_board(frame, side, "raw_cnn_board")
            raw_hsv = get_board(frame, side, "raw_hsv_board")
            if confirmed is None or raw_cnn is None or raw_hsv is None:
                continue

            total_stable_cells += 13 * 6

            # cell 履歴を更新
            for r in range(13):
                for c_col in range(6):
                    cnn_v = board_get(raw_cnn, r, c_col)
                    hsv_v = board_get(raw_hsv, r, c_col)
                    conf_v = board_get(confirmed, r, c_col)
                    # consensus (cnn==hsv かつ非 UNKNOWN)
                    if cnn_v != 10 and hsv_v != 10 and cnn_v == hsv_v:
                        cons_v = cnn_v
                    else:
                        cons_v = -1  # consensus 不明
                    key = (side, r, c_col)
                    cell_history[key].append((frame_idx, conf_v, cons_v))
                    if len(cell_history[key]) > 50:
                        cell_history[key] = cell_history[key][-50:]

            # corruption 検出
            corruptions = analyze_corruption_frame(frame, side)
            for corr in corruptions:
                r, c_col = corr["row"], corr["col"]
                conf_v = corr["confirmed_v"]
                kind = corr["kind"]
                kind_count[kind] += 1

                # 侵入経路分類
                cell_key = (side, r, c_col)
                hist_for_cell = cell_history.get(cell_key, [])
                pfc_hist_for_cell = pfc_hist.get(cell_key, [])

                # classify_route は (row,col) キーで渡す
                route = classify_route(
                    (r, c_col),
                    conf_v,
                    frame_idx,
                    {(r, c_col): hist_for_cell} if hist_for_cell else {},
                    {(r, c_col): pfc_hist_for_cell} if pfc_hist_for_cell else {},
                )
                route_count[route] += 1

                # サンプル記録
                if len(route_samples[route]) < MAX_SAMPLE_OUTPUT:
                    route_samples[route].append({
                        "file": path.name,
                        "frame_idx": frame_idx,
                        "t_sec": frame.get("t_sec"),
                        "side": side,
                        "row": r,
                        "col": c_col,
                        "kind": kind,
                        "confirmed": color_name(conf_v),
                        "consensus": color_name(corr["consensus_v"]),
                        "pfc_this_frame": corr["pfc_map"].get((r, c_col)),
                        "cfc_this_frame": corr["cfc_map"].get((r, c_col)),
                        "pfc_history_recent": pfc_hist_for_cell[-5:] if pfc_hist_for_cell else [],
                        "cell_history_recent": hist_for_cell[-10:] if hist_for_cell else [],
                    })

    # --- 結果表示 ---
    print(f"\n  STABLE フレーム数: {total_stable_frames}")
    print(f"  STABLE セル数 (延べ): {total_stable_cells}")
    total_corrupt = sum(kind_count.values())
    corrupt_rate = total_corrupt / max(total_stable_cells, 1) * 100
    print(f"  corruption 総件数: {total_corrupt} ({corrupt_rate:.3f}%)")

    print("\n  [種別分布]")
    for kind in ["color→color", "color→empty", "empty→color", "other"]:
        cnt = kind_count.get(kind, 0)
        pct = cnt / max(total_corrupt, 1) * 100
        print(f"    {kind:20s}: {cnt:6d} ({pct:5.1f}%)")

    print("\n  [侵入経路分布]")
    route_labels = {
        "a_infer_placement": "a: infer_placement/physics_fix 起源",
        "b_t2_freeze": "b: T2 フリーズ起源",
        "c_constraint_fill": "c: constraint_fill 起源",
        "d_other": "d: その他 / 不明",
    }
    for route_key, label in route_labels.items():
        cnt = route_count.get(route_key, 0)
        pct = cnt / max(total_corrupt, 1) * 100
        print(f"    {label}: {cnt:6d} ({pct:5.1f}%)")

    print("\n  [代表サンプル (経路別)]")
    for route_key, label in route_labels.items():
        samples = route_samples.get(route_key, [])
        if not samples:
            continue
        print(f"\n    --- {label} ---")
        for s in samples:
            print(f"      frame={s['frame_idx']} t={s['t_sec']:.2f}s "
                  f"{s['side']} ({s['row']},{s['col']}) "
                  f"{s['kind']} confirmed={s['confirmed']} consensus={s['consensus']}")
            if s["pfc_this_frame"] is not None:
                print(f"        physics_fix_this_frame new_val={color_name(s['pfc_this_frame'])}")
            if s["pfc_history_recent"]:
                print(f"        pfc_history_recent (frame,old,new): {s['pfc_history_recent']}")
            # cell_history_recent: (frame_idx, confirmed_v, consensus_v)
            hist_str = " / ".join(
                f"f{h[0]}:conf={color_name(h[1])},cons={color_name(h[2]) if h[2]>=0 else '?'}"
                for h in s["cell_history_recent"]
            )
            print(f"        cell_hist: {hist_str}")

    return {
        "file": path.name,
        "total_stable_cells": total_stable_cells,
        "total_corrupt": total_corrupt,
        "kind_count": dict(kind_count),
        "route_count": dict(route_count),
    }


def summarize(results: list[dict[str, Any]]) -> None:
    """全ファイルの合計集計を表示する."""
    print(f"\n{'='*60}")
    print("合計サマリ (全ファイル)")
    print(f"{'='*60}")

    total_cells = sum(r["total_stable_cells"] for r in results)
    total_corrupt = sum(r["total_corrupt"] for r in results)
    print(f"  STABLE セル数 (延べ合計): {total_cells}")
    print(f"  corruption 総件数: {total_corrupt} ({total_corrupt/max(total_cells,1)*100:.3f}%)")

    # 種別集計
    all_kind: dict[str, int] = defaultdict(int)
    all_route: dict[str, int] = defaultdict(int)
    for r in results:
        for k, v in r["kind_count"].items():
            all_kind[k] += v
        for k, v in r["route_count"].items():
            all_route[k] += v

    print("\n  [種別合計]")
    for kind in ["color→color", "color→empty", "empty→color", "other"]:
        cnt = all_kind.get(kind, 0)
        pct = cnt / max(total_corrupt, 1) * 100
        print(f"    {kind:20s}: {cnt:6d} ({pct:5.1f}%)")

    print("\n  [侵入経路合計]")
    route_labels = {
        "a_infer_placement": "a: infer_placement/physics_fix 起源",
        "b_t2_freeze": "b: T2 フリーズ起源",
        "c_constraint_fill": "c: constraint_fill 起源",
        "d_other": "d: その他 / 不明",
    }
    for route_key, label in route_labels.items():
        cnt = all_route.get(route_key, 0)
        pct = cnt / max(total_corrupt, 1) * 100
        print(f"    {label}: {cnt:6d} ({pct:5.1f}%)")

    # 仮説検証: a が支配的か?
    a_cnt = all_route.get("a_infer_placement", 0)
    b_cnt = all_route.get("b_t2_freeze", 0)
    print(f"\n  [仮説検証] 「infer_placement → T2 フリーズ」連鎖構造")
    print(f"    a (infer 起源): {a_cnt} 件")
    print(f"    b (T2 フリーズ): {b_cnt} 件")
    print(f"    注: b は a から派生するため a+b が実質的な infer 起源割合")
    ab_total = a_cnt + b_cnt
    ab_pct = ab_total / max(total_corrupt, 1) * 100
    print(f"    a+b 合計: {ab_total} 件 ({ab_pct:.1f}%)")


def main() -> None:
    """メイン処理."""
    root = Path(__file__).resolve().parent.parent
    results = []
    for rel_path in BOARD_LOG_FILES:
        path = root / rel_path
        if not path.exists():
            print(f"[WARN] ファイルが存在しない: {path}")
            continue
        result = analyze_file(path)
        results.append(result)

    if results:
        summarize(results)
    else:
        print("[ERROR] 解析可能なファイルがなかった")


if __name__ == "__main__":
    main()
