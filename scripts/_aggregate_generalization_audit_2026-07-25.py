"""汎化監査(2026-07-25) 全動画 ON/OFF 比較 summary_all.md 生成 (read-only集計)。

対象: scripts/_jobs_generalization_audit_2026-07-25.txt が
data/verify/generalization_audit_2026-07-25/ に書き出す
summary_{stem}_new.json / summary_{stem}_old.json を全動画分読み込み、
1 つの比較表 (summary_all.md) にまとめる。

新既定(new) = 着地色補正+Driftガード2種+試合境界フルクリア 全ON。
旧既定(old) = 上記4フラグ明示OFF。

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/_aggregate_generalization_audit_2026-07-25.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "generalization_audit_2026-07-25"
MANIFEST_PATH: Path = OUTPUT_DIR / "generalization_manifest.json"


def _load_manifest_stems() -> list[str]:
    """manifest から動画 stem 一覧を読む (存在しなければ出力dir内 summary から推測)。"""
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return [v["stem"] for v in manifest.get("videos", [])]
    stems: set[str] = set()
    for p in OUTPUT_DIR.glob("summary_*_new.json"):
        stem = p.stem[len("summary_"):-len("_new")]
        stems.add(stem)
    return sorted(stems, key=lambda s: (len(s), s))


def _load_summary(stem: str, tag: str) -> "dict | None":
    path = OUTPUT_DIR / f"summary_{stem}_{tag}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    # 単一動画実行時、summary.json のキーは video stem そのもの (例: "29")。
    return data.get(stem)


def _fmt_pct(v: "float | None") -> str:
    return f"{v:.1f}%" if v is not None else "N/A"


def _side_row(video: str, tag: str, side: str, stats: dict) -> str:
    corr = stats.get(f"stats_{side}_corroborated", {})
    drift = stats.get("drift_resync_suppressed", {})
    bb = stats.get("baseline_broken_stats", {})
    return (
        f"| {video} | {tag} | {side.upper()} "
        f"| {corr.get('n_events_total')} "
        f"| {corr.get('n_never_reflected')} "
        f"| {_fmt_pct(corr.get('effective_pct_within_acceptance_8f'))} "
        f"| {corr.get('delay_frames_median')} "
        f"| {drift.get(f'start_guard_suppressed_{side}', 0)} "
        f"| {drift.get(f'hsv_gate_suppressed_{side}', 0)} "
        f"| {bb.get(f'reset_count_{side}', 0)} |"
    )


def _build_table(stems: list[str]) -> list[str]:
    lines = [
        "| video | config | side | n_events | 未反映 | 8f実効達成率 | delay中央値 | "
        "drift_start_guard抑制 | drift_hsv_gate抑制 | baseline_broken発火 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    n_missing: list[str] = []
    for stem in stems:
        for tag in ("new", "old"):
            data = _load_summary(stem, tag)
            if data is None:
                n_missing.append(f"video_{stem}_{tag}")
                continue
            for side in ("1p", "2p"):
                lines.append(_side_row(f"video_{stem}", tag, side, data))
    if n_missing:
        lines.append("")
        lines.append(f"欠損 (未完走の可能性): {', '.join(n_missing)}")
    return lines


def main() -> None:
    stems = _load_manifest_stems()
    lines = [
        "# 汎化監査 (2026-07-25) 新既定(4修正ON) vs 旧既定(4フラグ明示OFF) 比較",
        "",
        f"対象動画: {len(stems)} 本 (調整に未使用、詳細は generalization_manifest.json)",
        "",
    ]
    lines.extend(_build_table(stems))
    out_path = OUTPUT_DIR / "summary_all.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] 出力: {out_path}")


if __name__ == "__main__":
    main()
