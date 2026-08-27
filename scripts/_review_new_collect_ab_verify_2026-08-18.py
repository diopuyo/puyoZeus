"""新盤面収集方式 (move-segmented + physics-persistence) の自己検収用A/B集計 (2026-08-18)。

検収レビュア (self-verify) が「指定シーン全時間帯」を数値で突合するための
計装専用スクリプト。src/・collect_boards_lean.py は一切変更しない。

比較対象 (data/verify/move_segment_physics_filter_2026-08-18/、コーダエージェント
生成の既存npz。commit 9fba9ac/97cc37f の実測値そのものを再生成したファイルで
あることを row 数・t_sec範囲の突合で確認済み):
  {vid}_baseline.npz  : 本番採用フラグのみ (新2フラグ=OFF、旧方式相当)
  {vid}_new_v2or.npz  : 本番採用フラグ + --enable-move-segmented-recording
                        --enable-physics-persistence-filter (OR条件化後、最終構成)
  いずれも --start-sec 150 --max-sec 150 --with-next --enable-phantom-board-guard

出力:
  1. 行数比 (new_v2or/baseline)  — commit 97cc37f の実測 (88.7%等) の再現確認
  2. 「1手に1枚」の全時間帯チェック: tsumo_count 増分 (=手数) に対する
     new_v2or npz 記録行数の比率をside別に算出 (全区間、スポット確認でない)
  3. 同一 game_idx 内での grid 重複行 (多重記録) の有無を全数走査
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EVIDENCE_DIR = Path("data/verify/move_segment_physics_filter_2026-08-18")
TARGET_VIDEOS: tuple[str, ...] = ("36", "52", "c100")
NEW_MODE = "new_v2or"
OUT_PATH = EVIDENCE_DIR / "ab_verify_report_review_2026-08-18.json"


def _load(vid: str, mode: str) -> "dict | None":
    p = EVIDENCE_DIR / f"{vid}_{mode}.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def _moves_per_side(d: dict) -> dict:
    """side別のtsumo_count増分合計 (=手数の推定値) を返す。"""
    side = d["side"]
    tsumo = d["tsumo_count"]
    game_idx = d["game_idx"]
    t_sec = d["t_sec"]
    out: dict[str, int] = {}
    for s in np.unique(side):
        m = side == s
        order = np.argsort(t_sec[m])
        tt = tsumo[m][order]
        gg = game_idx[m][order]
        moves = 0
        for i in range(1, len(tt)):
            if gg[i] == gg[i - 1] and tt[i] > tt[i - 1]:
                moves += int(tt[i] - tt[i - 1])
        out[str(s)] = moves
    return out


def _duplicate_grid_rows(d: dict) -> int:
    """同一 (side, game_idx) 内で grid が完全一致する行のペア数 (多重記録候補)。"""
    side = d["side"]
    game_idx = d["game_idx"]
    grids = d["grids"]
    groups: dict[tuple, list[int]] = {}
    for i in range(len(grids)):
        key = (str(side[i]), int(game_idx[i]))
        groups.setdefault(key, []).append(i)
    n_dup = 0
    for idxs in groups.values():
        seen = set()
        for i in idxs:
            b = grids[i].tobytes()
            if b in seen:
                n_dup += 1
            seen.add(b)
    return n_dup


def main() -> None:
    report: dict[str, dict] = {}
    total_base = 0
    total_new = 0
    for vid in TARGET_VIDEOS:
        base = _load(vid, "baseline")
        new = _load(vid, NEW_MODE)
        entry: dict = {}
        if base is None or new is None:
            entry["status"] = "pending"
            report[vid] = entry
            continue
        n_base = len(base["grids"])
        n_new = len(new["grids"])
        total_base += n_base
        total_new += n_new
        entry["n_baseline"] = n_base
        entry["n_new_v2or"] = n_new
        entry["ratio_new_over_baseline_pct"] = round(n_new / n_base * 100, 1) if n_base else None
        entry["t_range_baseline"] = [float(base["t_sec"].min()), float(base["t_sec"].max())]
        entry["t_range_new_v2or"] = [float(new["t_sec"].min()), float(new["t_sec"].max())]
        entry["moves_baseline"] = _moves_per_side(base)
        entry["moves_new_v2or"] = _moves_per_side(new)
        n_new_by_side = {s2: int((new["side"] == s2).sum()) for s2 in np.unique(new["side"])}
        entry["n_new_v2or_by_side"] = n_new_by_side
        entry["boards_per_move_new_v2or"] = {
            s: round(n / entry["moves_new_v2or"][s], 3) if entry["moves_new_v2or"].get(s) else None
            for s, n in n_new_by_side.items()
        }
        entry["duplicate_grid_rows_new_v2or"] = _duplicate_grid_rows(new)
        entry["duplicate_grid_rows_baseline"] = _duplicate_grid_rows(base)
        report[vid] = entry
    report["_total"] = {
        "n_baseline": total_base, "n_new_v2or": total_new,
        "ratio_pct": round(total_new / total_base * 100, 1) if total_base else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
