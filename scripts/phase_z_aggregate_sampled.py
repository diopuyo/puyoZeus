"""ユーザーの sampled_review レビュー結果を集計し動画別・全体の真 accuracy を計算。"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ユーザー入力 (各動画 20 cells、行順=動画順=sampled.csv の動画 ID 昇順)
USER_REVIEW: dict[str, str] = {
    "v01": "RGYEGGGYBBBGGBGYRRYE",
    "v02": "YGEBYBEEEEGEEEOEEYEE",
    "v03": "EERGEEPEEEEEEREEEEEE",
    "v04": "BRYYYYYYYRYYYYYGYYYY",
    "v05": "GBEGBEEGEEPEEERBRBPE",
    "v06": "YYBYYRBBYBREEEEEEEEE",
    "v07": "RRYBRGGYRYREBYBYRYYR",
    "v08": "YRBRYYYPRYYPYYRRPYOY",
    "v09": "BBPEEEEGEGBEPEPBBEEE",
    "v10": "EEEEEEEPRPYEEEEEPGRR",
    "v11": "REBEPYPBPBRYRYEPEEYP",
    "v12": "PYGYYYPRYYYBRYYRYRYE",
    "v13": "YRRYEEEEPEYPRBBBREYR",
    "v14": "YPGBGYBGGBYYPBEPGYGB",
    "v15": "EBRPGGPOPRRGGGPRPGPP",
    "v16": "GGPGBBGOGGYEEEEEEYGY",
    "v17": "GPRYPPYEEEEEEEEEEEEE",
    "v19": "BBYBBGGPGGGGBGGYGGGY",
}

# エフェクト・移動中・UI 由来 cell の除外 (動画別、1-indexed cell インデックス)
EXCLUDED: dict[str, set[int]] = {
    "v02": {6, 16, 17, 19, 20},   # 6=移動中、16/17/19=エフェクト、20=X印
    "v05": {2},                    # 2=エフェクト
    "v08": {15, 17, 18, 19},       # エフェクト
    "v11": {18},                    # エフェクト
    "v16": {12, 13, 14},            # エフェクト
}

CHAR_TO_LABEL = {
    "E": "EM", "R": "RED", "B": "BLUE", "G": "GRN",
    "Y": "YEL", "P": "PUR", "O": "OJM", "?": "??",
}

# cross_video の hard violations 数 (各動画 30s 区間、推定 acc 計算用)
HARD_COUNTS: dict[str, int] = {
    "v01": 83, "v02": 131, "v03": 133, "v04": 199, "v05": 123,
    "v06": 115, "v07": 77, "v08": 111, "v09": 121, "v10": 178,
    "v11": 122, "v12": 87, "v13": 83, "v14": 58, "v15": 89,
    "v16": 94, "v17": 115, "v19": 146,
}
def get_eval_total_from_labels(vid: str) -> int:
    """labels.csv から is_chain=0 の cell 数を集計 (正確な評価対象数)。"""
    cross_dir = _ROOT / "data/verify/phase_z_review/cross_video"
    for d in cross_dir.iterdir():
        if not d.is_dir() or not d.name.startswith(f"{vid}_m_"):
            continue
        # 30s 区間のみ
        parts = d.name.split("_")
        try:
            if int(parts[3]) - int(parts[2]) < 20:
                continue
        except (IndexError, ValueError):
            continue
        labels = d / "labels.csv"
        if not labels.exists():
            continue
        n = 0
        with labels.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("is_chain") == "0":
                    n += 1
        return n
    return 3500  # フォールバック


def main() -> int:
    csv_path = (
        _ROOT
        / "data/verify/phase_z_review/sampled_review/sampled.csv"
    )
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return 1

    # CSV を動画別にグループ化 (sampled.csv の global_id 順 = 動画順)
    rows_by_video: dict[str, list[dict]] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # video カラムは "v01_m_259_289" のような形式
            vname = r["video"]
            vid = vname.split("_")[0]  # "v01"
            rows_by_video.setdefault(vid, []).append(r)

    # 動画ごとに集計
    print(f"{'video':<5} {'total':<6} {'eval':<6} {'match':<6} "
          f"{'mm':<4} {'excl':<5} {'sample acc':<11} "
          f"{'真 acc 推定':<12}")
    print("-" * 72)
    summary: list[dict] = []
    overall_match = 0
    overall_mm = 0
    overall_excl = 0
    overall_eval = 0
    overall_total = 0
    for vid, user_str in USER_REVIEW.items():
        rows = rows_by_video.get(vid, [])
        if not rows:
            print(f"{vid:<5} ERROR: rows not found")
            continue
        if len(user_str) != len(rows):
            print(f"{vid:<5} WARN: user入力 {len(user_str)} != rows {len(rows)}")
        excluded_set = EXCLUDED.get(vid, set())
        match_n = 0
        mm_n = 0
        excl_n = 0
        for i, r in enumerate(rows):
            sheet_idx = i + 1  # 1-indexed
            if sheet_idx in excluded_set:
                excl_n += 1
                continue
            if i >= len(user_str):
                continue
            user_label = CHAR_TO_LABEL.get(user_str[i].upper())
            if user_label is None:
                continue
            rec = r["recognized"]
            if rec == user_label:
                match_n += 1
            else:
                mm_n += 1
        eval_n = match_n + mm_n
        total = len(rows)
        sample_acc = (match_n / eval_n) if eval_n > 0 else 0.0
        # 真 acc 推定: 評価対象全 cell のうち hard violation の比率は
        # (mm / eval) * (hard / hard) ≒ サンプル誤り率を hard 違反の真陽性率として外挿
        hard_n = HARD_COUNTS.get(vid, 0)
        eval_total_cells = get_eval_total_from_labels(vid)
        # サンプル中の真誤り率 = mm / eval
        # → hard violations のうち真誤り = hard * (mm / eval)
        # → 真 acc = 1 - (hard * mm / eval) / eval_total_cells
        if eval_n > 0:
            true_errors = hard_n * (mm_n / eval_n)
            true_acc = 1 - true_errors / eval_total_cells
        else:
            true_acc = 0.0
        summary.append({
            "video": vid, "total": total, "eval": eval_n,
            "match": match_n, "mm": mm_n, "excl": excl_n,
            "sample_acc": sample_acc * 100,
            "true_acc": true_acc * 100,
        })
        print(
            f"{vid:<5} {total:<6} {eval_n:<6} {match_n:<6} {mm_n:<4} "
            f"{excl_n:<5} {sample_acc * 100:<10.2f}% "
            f"{true_acc * 100:<11.3f}%"
        )
        overall_match += match_n
        overall_mm += mm_n
        overall_excl += excl_n
        overall_eval += eval_n
        overall_total += total
    print("-" * 72)
    sample_acc_all = (
        overall_match / overall_eval if overall_eval > 0 else 0.0
    )
    print(
        f"{'ALL':<5} {overall_total:<6} {overall_eval:<6} "
        f"{overall_match:<6} {overall_mm:<4} {overall_excl:<5} "
        f"{sample_acc_all * 100:<10.2f}%"
    )
    print()
    print(f"全 sampling: {overall_eval} cells、真 match {overall_match}、"
          f"真誤り {overall_mm} (sample acc {sample_acc_all * 100:.2f}%)")
    print()
    # 動画別 mismatch 詳細
    for s in summary:
        if s["mm"] == 0:
            continue
        vid = s["video"]
        rows = rows_by_video[vid]
        user_str = USER_REVIEW[vid]
        excluded_set = EXCLUDED.get(vid, set())
        print(f"\n=== {vid} mismatch ({s['mm']}) ===")
        for i, r in enumerate(rows):
            sheet_idx = i + 1
            if sheet_idx in excluded_set:
                continue
            if i >= len(user_str):
                continue
            user_label = CHAR_TO_LABEL.get(user_str[i].upper())
            rec = r["recognized"]
            if rec != user_label:
                print(
                    f"  #{sheet_idx} t={r['time']} "
                    f"{r['side']}r{r['row']}c{r['col']} | "
                    f"{rec:5s} -> {user_label:5s} | {r['reasons'][:60]}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
