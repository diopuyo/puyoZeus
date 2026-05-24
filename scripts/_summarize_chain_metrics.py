"""cycle 46 全 8 動画の chain 系 metric サマリ."""
from __future__ import annotations
import json
from pathlib import Path


def main():
    videos = ["v29m2", "v40m7", "v51m2", "v57m2", "v70m2", "v89m3", "v95m15", "v97m11"]
    print(f"{'video':<10}{'critical':>10}{'no_puyo_loss':>15}{'retro_chain_missing':>22}{'too_short':>12}{'chain_no_disappear':>22}")
    print("-" * 95)
    for v in videos:
        path = Path(f"data/verify/cycle46_eval/cycle46_{v}_buf15.json")
        if not path.is_file():
            continue
        d = json.load(open(path, encoding="utf-8"))
        s = d.get("summary", {})
        bm = s.get("by_metric_critical", {})
        print(
            f"{v:<10}"
            f"{s.get('critical', 0):>10}"
            f"{bm.get('chain_no_puyo_loss', 0):>15}"
            f"{bm.get('retrospective_chain_missing', 0):>22}"
            f"{bm.get('chain_state_too_short', 0):>12}"
            f"{bm.get('chain_no_disappear', 0):>22}"
        )


if __name__ == "__main__":
    main()
