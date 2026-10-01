"""既存採点結果・表示差・入力ハッシュをマージン実験の比較原票にまとめる。"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.measure_margin_clock import ROOT, RECORDS
from src.production_config import exchange_event_flags
from src.scoring import compute_effective_rate

SOURCES = ("q_7gc4TgFig", "review", "fcXG83vInDY", "mia8KCjr52g", "zenchi")
EXAMPLE_LIMIT = 10
REFERENCE = Path("/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/pending_expiry/e36b_on")


def read(path: Path) -> Any:
    """UTF-8原票を変更せず読む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """記録と出力を内容ハッシュで識別する。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def directory(root: Path, source: str) -> Path:
    """既存採点器のディレクトリ構造に揃える。"""
    suffix = source if source in ("review", "zenchi") else f"renders/{source}/on"
    return root / "on" / suffix


def rate_context(display: Any, clocks: list[dict]) -> dict:
    """現在時刻のレートが違う表示行を、表示由来別に数える（発火時計とは別）。"""
    sources: Counter[str] = Counter()
    for row in clocks:
        old, new = row["old_origin"], row["first_placement"]
        if old is None or new is None:
            continue
        mask = display["game_idx"] == row["game"]
        stamps = display["t_sec"][mask]
        changed = np.array([compute_effective_rate(max(0., t-old)) !=
                            compute_effective_rate(max(0., t-new)) for t in stamps], dtype=bool)
        sources.update(display["source"][mask][changed].tolist())
    return dict(display_frames=sum(sources.values()), sources=dict(sources))


def differences(source: str) -> dict:
    """同一行・同一母数の表示と平滑前確率を比較する。"""
    before = directory(ROOT / "off", source)
    after = directory(ROOT / "on", source)
    clock = read(after / "margin_origins.json")
    columns = {}
    with np.load(before / "display.npz") as off, np.load(after / "display.npz") as on:
        for name in ("t_sec", "game_idx", "state1", "state2", "score1", "score2"):
            np.testing.assert_array_equal(off[name], on[name])
        for name in ("display_adv", "display_p1", "source"):
            left, right = off[name], on[name]
            changed = left != right
            numeric = np.issubdtype(left.dtype, np.number)
            if numeric:
                changed &= ~(np.isnan(left) & np.isnan(right))
            delta = np.abs(left-right) if numeric else None
            columns[name] = dict(changed=int(changed.sum()), frames=len(left),
                max_abs=float(np.nanmax(delta)) if numeric else None,
                first_times=off["t_sec"][changed][:EXAMPLE_LIMIT].tolist())
        context = rate_context(on, clock)
    return dict(columns=columns, games=clock,
                rate_context=context,
                rate_difference_frames=sum(r["applied_difference_frames"] for r in clock))


def actual_display(variant: str) -> dict:
    """既存の平滑化検収器から、実表示の場面時刻と補助qを取得する。"""
    from scripts import report_switch_smoothing as report
    from scripts.report_e35 import SCENE_STAMP
    report.OUT_ROOT = ROOT
    data = report.load(variant, "review")
    index = int(np.argmin(np.abs(data["t_sec"]-SCENE_STAMP)))
    return dict(scene_first_sec=report.scene_first_sec(variant),
                stamp=float(data["t_sec"][index]),
                p2_display=float(1-report.shown_probability(data)[index]),
                q=report.display_q(variant))


def main() -> None:
    """再計算は行わず、採点済み3条件の数値と証拠を保存する。"""
    summaries = {v: read(ROOT / v / "SUMMARY.json") for v in ("legacy_off", "off", "on")}
    equivalence = {}
    for source in SOURCES:
        actual = directory(ROOT / "legacy_off", source) / "display.npz"
        reference = directory(REFERENCE, source) / "display.npz"
        equivalence[source] = dict(actual=digest(actual), reference=digest(reference),
                                   identical=actual.read_bytes() == reference.read_bytes())
    value = dict(summaries=summaries, baseline_display_equivalence=equivalence,
        differences={s: differences(s) for s in SOURCES},
        production_flags=exchange_event_flags(),
        record_sha256={s: digest(RECORDS / f"{s}.jsonl.gz") for s in SOURCES},
        production_config_sha256=digest(Path("src/production_config.py")),
        actual_display={v: actual_display(v) for v in summaries},
        scenes={v: read(ROOT / v / "SCENE.json") for v in summaries})
    (ROOT / "COMPARISON.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({s: value["differences"][s]["columns"] for s in SOURCES}), flush=True)


if __name__ == "__main__":
    main()
