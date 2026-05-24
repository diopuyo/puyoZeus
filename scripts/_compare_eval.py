"""HSV-only / CNN-only / per-video model の 3-way 比較表 (一時)."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(p: Path) -> dict:
    if not p.exists():
        return {}
    with p.open() as f:
        return {r["video"]: r for r in csv.DictReader(f, delimiter="\t")}


def main() -> None:
    hsv = load(ROOT / "data/phase_b_eval_summary.tsv")
    pv = load(ROOT / "data/phase_b_eval_summary_pv.tsv")
    sm3 = load(ROOT / "data/phase_b_eval_summary_smooth3.tsv")
    pv2 = load(ROOT / "data/phase_b_eval_summary_pv2.tsv")
    if pv2:
        keys = sorted(hsv.keys() & pv.keys() & pv2.keys())
        print(
            f"{'video':<5} | {'HSV':>6} | {'PV':>6} | "
            f"{'PV+sm3':>6} | {'PV2':>6} | best"
        )
        print("-" * 60)
        for v in keys:
            h_s = float(hsv[v]["1P_stable_pct"])
            p_s = float(pv[v]["1P_stable_pct"])
            s_s = float(sm3[v]["1P_stable_pct"]) if v in sm3 else 0.0
            p2_s = float(pv2[v]["1P_stable_pct"])
            cands = {"HSV": h_s, "PV": p_s, "SM3": s_s, "PV2": p2_s}
            best = max(cands, key=lambda k: cands[k])
            print(
                f"{v:<5} | {h_s:>5.1f}% | {p_s:>5.1f}% | "
                f"{s_s:>5.1f}% | {p2_s:>5.1f}% | {best}"
            )

        def avg(d: dict, key: str) -> float:
            return sum(float(r[key]) for r in d.values()) / len(d)
        print("-" * 60)
        print(
            f"AVG STABLE: HSV {avg(hsv, '1P_stable_pct'):.1f}%  "
            f"PV {avg(pv, '1P_stable_pct'):.1f}%  "
            f"SM3 {avg(sm3, '1P_stable_pct'):.1f}%  "
            f"PV2 {avg(pv2, '1P_stable_pct'):.1f}%"
        )
        print(
            f"AVG drift:  HSV {avg(hsv, '1P_drift'):.1f}  "
            f"PV {avg(pv, '1P_drift'):.1f}  "
            f"SM3 {avg(sm3, '1P_drift'):.1f}  "
            f"PV2 {avg(pv2, '1P_drift'):.1f}"
        )
        return
    has_sm3 = bool(sm3)
    if has_sm3:
        keys = sorted(hsv.keys() & pv.keys() & sm3.keys())
    else:
        cnn = load(ROOT / "data/phase_b_eval_summary_v1.tsv")
        has_pv = bool(pv)
        keys = sorted(hsv.keys() & cnn.keys() & (pv.keys() if has_pv else hsv.keys()))
    if has_sm3:
        print(
            f"{'video':<5} | {'HSV stbl/drf':>14} | "
            f"{'PV stbl/drf':>14} | {'PV+sm3 stbl/drf':>16} | best"
        )
        print("-" * 70)
        for v in keys:
            h_s = float(hsv[v]["1P_stable_pct"])
            p_s = float(pv[v]["1P_stable_pct"])
            s_s = float(sm3[v]["1P_stable_pct"])
            best = "SM3" if s_s >= max(h_s, p_s) else ("PV" if p_s >= h_s else "HSV")
            print(
                f"{v:<5} | {hsv[v]['1P_stable_pct']:>5}%/{hsv[v]['1P_drift']:>3}    "
                f"| {pv[v]['1P_stable_pct']:>5}%/{pv[v]['1P_drift']:>3}    "
                f"| {sm3[v]['1P_stable_pct']:>5}%/{sm3[v]['1P_drift']:>3}      | {best}"
            )

        def avg(d: dict, key: str) -> float:
            return sum(float(r[key]) for r in d.values()) / len(d)
        print("-" * 70)
        print(
            f"AVG STABLE: HSV {avg(hsv, '1P_stable_pct'):.1f}%  "
            f"PV {avg(pv, '1P_stable_pct'):.1f}%  "
            f"PV+sm3 {avg(sm3, '1P_stable_pct'):.1f}%"
        )
        print(
            f"AVG drift:  HSV {avg(hsv, '1P_drift'):.1f}  "
            f"PV {avg(pv, '1P_drift'):.1f}  "
            f"PV+sm3 {avg(sm3, '1P_drift'):.1f}"
        )
        return  # SM3 ありなら従来比較は省略

    if has_pv:
        header = (
            f"{'video':<5} | {'HSV stbl/drf/rsy':>16} | "
            f"{'CNN stbl/drf/rsy':>16} | "
            f"{'PV stbl/drf/rsy':>16} | best"
        )
    else:
        header = (
            f"{'video':<5} | {'HSV stbl/drf/rsy':>16} | "
            f"{'CNN stbl/drf/rsy':>16} | diff"
        )
    print(header)
    print("-" * len(header))

    for v in keys:
        h = hsv[v]
        c = cnn[v]
        h_s = float(h["1P_stable_pct"])
        c_s = float(c["1P_stable_pct"])
        line = (
            f"{v:<5} | "
            f"{h['1P_stable_pct']:>5}%/{h['1P_drift']:>3}/{h['1P_resync']:>2}      "
            f"| {c['1P_stable_pct']:>5}%/{c['1P_drift']:>3}/{c['1P_resync']:>2}      "
        )
        if has_pv:
            p = pv[v]
            p_s = float(p["1P_stable_pct"])
            best = (
                "PV" if p_s >= max(h_s, c_s)
                else "CNN" if c_s >= h_s else "HSV"
            )
            line += (
                f"| {p['1P_stable_pct']:>5}%/{p['1P_drift']:>3}/{p['1P_resync']:>2}      | {best}"
            )
        else:
            line += f"| {c_s - h_s:+.1f}pt"
        print(line)

    print("-" * len(header))

    def avg(d: dict, key: str) -> float:
        return sum(float(r[key]) for r in d.values()) / len(d)

    print(
        f"AVG STABLE: HSV {avg(hsv, '1P_stable_pct'):.1f}%  "
        f"CNN {avg(cnn, '1P_stable_pct'):.1f}%  "
        + (f"PV {avg(pv, '1P_stable_pct'):.1f}%" if has_pv else "")
    )
    print(
        f"AVG drift:  HSV {avg(hsv, '1P_drift'):.1f}  "
        f"CNN {avg(cnn, '1P_drift'):.1f}  "
        + (f"PV {avg(pv, '1P_drift'):.1f}" if has_pv else "")
    )
    print(
        f"AVG resync: HSV {avg(hsv, '1P_resync'):.1f}  "
        f"CNN {avg(cnn, '1P_resync'):.1f}  "
        + (f"PV {avg(pv, '1P_resync'):.1f}" if has_pv else "")
    )


if __name__ == "__main__":
    main()
