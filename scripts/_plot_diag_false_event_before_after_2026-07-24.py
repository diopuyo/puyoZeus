"""修正D(2026-07-24) viz: 既存の diag records.json (before/after) からタイムラインPNGを描く。

再simulateは行わない(既に保存済みの
data/verify/diag_false_event_source_2026-07-24/records.json (before, flag=False) と
data/verify/diag_false_event_source_2026-07-24_after/records.json (after, flag=True)
を読み込んで描画するだけの軽量スクリプト)。

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/_plot_diag_false_event_before_after_2026-07-24.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
BEFORE_PATH = (
    PROJ_ROOT / "data" / "verify" / "diag_false_event_source_2026-07-24" / "records.json"
)
AFTER_PATH = (
    PROJ_ROOT / "data" / "verify" / "diag_false_event_source_2026-07-24_after" / "records.json"
)
OUTPUT_DIR = (
    PROJ_ROOT / "data" / "verify" / "chain_formula_simulate_verify_before_after_2026-07-24"
)


def _write_timeline_png(
    before: list[dict], after: list[dict], video_stem: str, out_path: Path,
) -> None:
    """before(検証OFF) / after(検証ON) の trigger タイムラインを並べて描く。

    色分け: 緑=real (chain_count_resimulated>=1) / 赤=false (疑似発火)。
    形状: 丸=chain_tracker (物理的お邪魔ぷよ減少検知、地に足がついた真値) /
          三角=early_fire_synthetic (機能B/D 早期発火)。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
    for ax, records, label in (
        (axes[0], before, "before (enable_chain_formula_simulate_verify=False, 既定)"),
        (axes[1], after, "after (enable_chain_formula_simulate_verify=True)"),
    ):
        for r in records:
            color = "red" if r["is_false"] else "green"
            marker = "o" if r["source"] == "chain_tracker" else "^"
            y = 1.0 if r["side"] == "1P" else 0.0
            ax.scatter(r["trigger_sec"], y, color=color, marker=marker, s=70)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["2P", "1P"])
        ax.set_ylim(-0.5, 1.5)
        n_false = sum(1 for r in records if r["is_false"])
        ax.set_title(f"{label}  (総{len(records)}件、偽={n_false}件)")
        ax.grid(axis="x", alpha=0.3)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=10, label="real / chain_tracker (物理ground truth)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="green", markersize=10, label="real / early_fire_synthetic"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="red", markersize=10, label="false / early_fire_synthetic (疑似発火)"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (sec)")
    fig.suptitle(f"{video_stem}: chain trigger before/after 機能D simulate_verify 比較 (2026-07-24)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    before_all = json.loads(BEFORE_PATH.read_text(encoding="utf-8"))
    after_all = json.loads(AFTER_PATH.read_text(encoding="utf-8"))

    videos = sorted({r["video"] for r in before_all} | {r["video"] for r in after_all})
    summary: dict = {"videos": {}}
    out_paths: list[str] = []
    for stem in videos:
        before = [r for r in before_all if r["video"] == stem]
        after = [r for r in after_all if r["video"] == stem]
        n_before_false = sum(1 for r in before if r["is_false"])
        n_after_false = sum(1 for r in after if r["is_false"])
        summary["videos"][stem] = {
            "n_before_total": len(before), "n_before_false": n_before_false,
            "n_before_real": len(before) - n_before_false,
            "n_after_total": len(after), "n_after_false": n_after_false,
            "n_after_real": len(after) - n_after_false,
        }
        out_path = OUTPUT_DIR / f"timeline_{stem}.png"
        _write_timeline_png(before, after, stem, out_path)
        out_paths.append(str(out_path))
        print(f"{stem}: before={len(before)}(偽{n_before_false}) -> after={len(after)}(偽{n_after_false})  => {out_path}")

    (OUTPUT_DIR / "plot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n[PNG paths]")
    for p in out_paths:
        print(" ", p)


if __name__ == "__main__":
    main()
