"""cycle 46 全 8 動画 buf15s vs 既存 (バッファなし) 比較."""
from __future__ import annotations
import json
from pathlib import Path


def load(p):
    if not p.is_file(): return {}
    return json.load(open(p, encoding="utf-8"))


def main():
    videos = ["v29m2", "v40m7", "v51m2", "v57m2", "v70m2", "v89m3", "v95m15", "v97m11"]
    print("=" * 80)
    print("cycle 46 全 8 動画 (= 15s buffer) vs 既存 (= no buffer) 比較")
    print("=" * 80)
    print()
    print(f"{'video':<10} {'old crit':>10} {'new crit':>10} {'diff':>10}")
    print("-" * 50)
    old_total = 0
    new_total = 0
    diffs = {}
    for v in videos:
        d_new = load(Path(f"data/verify/cycle46_eval/cycle46_{v}_buf15.json"))
        # 既存 = cycle 33 (= baseline 推論軸あり) で比較
        # ただし v89m3, v97m11, v70m2 のみ cycle 33 ある、 他は eval なし
        # 比較用に baseline (= cycle 32d/e/g 系の元) 既存 8 動画 viz 評価値を使う
        # 代替: v89m3/v97m11/v70m2 は cycle 33、 他は計測してないので NA
        candidate_old = None
        for source_dir in ("cycle33_eval", "cycle37_eval"):
            p = Path(f"data/verify/{source_dir}/cycle33_{v}.json")
            if p.is_file():
                candidate_old = p
                break
            p = Path(f"data/verify/{source_dir}/cycle37_{v}.json")
            if p.is_file():
                candidate_old = p
                break
        d_old = load(candidate_old) if candidate_old else {}
        s_old = d_old.get("summary", {})
        s_new = d_new.get("summary", {})
        old_c = s_old.get("critical", 0)
        new_c = s_new.get("critical", 0)
        diff = new_c - old_c
        diffs[v] = (old_c, new_c, diff)
        old_total += old_c
        new_total += new_c
        print(f"{v:<10} {old_c:>10} {new_c:>10} {diff:>+10}")
    print("-" * 50)
    print(f"{'合計':<10} {old_total:>10} {new_total:>10} {new_total-old_total:>+10}")

    # 主要メトリクス別差分 (= 全動画合計)
    print()
    print("=== メトリクス別 全動画合計 差分 ===")
    all_metrics = set()
    metric_old = {}
    metric_new = {}
    for v in videos:
        d_new = load(Path(f"data/verify/cycle46_eval/cycle46_{v}_buf15.json"))
        candidate_old = Path(f"data/verify/cycle33_eval/cycle33_{v}.json")
        if not candidate_old.is_file():
            continue
        d_old = load(candidate_old)
        s_old = d_old.get("summary", {}).get("by_metric_critical", {})
        s_new = d_new.get("summary", {}).get("by_metric_critical", {})
        for m, c in s_old.items():
            metric_old[m] = metric_old.get(m, 0) + c
            all_metrics.add(m)
        for m, c in s_new.items():
            metric_new[m] = metric_new.get(m, 0) + c
            all_metrics.add(m)
    print(f"{'metric':<35} {'old':>8} {'new':>8} {'diff':>8}")
    print("-" * 65)
    for m in sorted(all_metrics):
        o = metric_old.get(m, 0)
        n = metric_new.get(m, 0)
        print(f"{m:<35} {o:>8} {n:>8} {n-o:>+8}")


if __name__ == "__main__":
    main()
