"""修正後の5記録再生を本番登録時と照合し、固定ラベルで再採点する。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import aggregate_e3_exchange_eval_20260926 as metrics
from scripts import report_e10b_exchange_20260927 as labels
from scripts.replay_exchange_event_20260926 import compare
from scripts.run_e3_exchange_eval_20260926 import SOURCES

RECORD_SOURCES = (*SOURCES, "review", "zenchi")
BASELINE = Path("logs/pending_expiry/e36b_on/on")
REGISTERED = Path("/mnt/d/puyo_analyzer/wt_switch/logs/switch_smoothing/cli_prod")
RAW_COLUMNS = ("display_p1", "source", "adv_raw_last", "t_sec", "game_idx")


def baseline_directory(source: str) -> Path:
    """既存検収物の配置を解決する。"""
    return BASELINE / source if source in ("review", "zenchi") else BASELINE / "renders" / source / "on"


def equivalence(root: Path, registered: Path) -> dict:
    """全ファイルバイトと平滑前列の同一性を、5記録すべてで検証する。"""
    rows = {}
    for source in RECORD_SOURCES:
        actual = root / source
        result = compare(registered / source, actual)
        with np.load(actual / "display.npz") as new, np.load(baseline_directory(source) / "display.npz") as old:
            for name in RAW_COLUMNS:
                assert new[name].dtype == old[name].dtype
                assert new[name].tobytes() == old[name].tobytes(), (source, name)
        hashes = {name: hashlib.sha256((actual / name).read_bytes()).hexdigest()
                  for name in ("display.npz", "events.jsonl")}
        rows[source] = dict(result, raw_columns="byte_identical", sha256=hashes)
    return rows


def death_metrics(root: Path) -> dict:
    """既存採点器の37件と同じ母集団・補完ラベルで誤確定を数える。"""
    games = labels.read(labels.ZENCHI / "official_games.json")
    rows = labels.death_rows(root / "zenchi", "zenchi", games)
    panel = labels.read(Path("logs/e10b/panel_outcomes.json"))
    for source in SOURCES:
        windows, _ = metrics.outcomes(source)
        seconds = [dict(start=w["start"] / metrics.FPS, end=w["end"] / metrics.FPS,
                        winner=w["winner"]) for w in windows]
        rows.extend(labels.death_rows(root / source, source, seconds))
    for row in rows:
        if row["winner"] is None:
            match = next((r for r in panel if r["source"] == row["source"]
                          and r["game_idx"] == row["game_idx"]), None)
            if match is not None and match["winner"] is not None:
                row.update(winner=match["winner"], false_positive=match["winner"] == row["side"])
    return dict(false=sum(r["false_positive"] is True for r in rows), total=len(rows),
                unlabelled=sum(r["winner"] is None for r in rows))


def verify(root: Path, registered: Path) -> dict:
    """平滑化ONとE36bの表示採点を分け、既知の1フレーム差を隠さず残す。"""
    equal = equivalence(root, registered)
    with np.load(root / SOURCES[0] / "display.npz") as display:
        q = metrics.m3_scores(display, metrics.outcomes(SOURCES[0])[0])["groups"]["all"]
    games = labels.read(labels.ZENCHI / "official_games.json")
    zenchi = labels.agreement(root / "zenchi/display.npz", games)
    baseline_zenchi = labels.agreement(baseline_directory("zenchi") / "display.npz", games)
    deaths = death_metrics(root)
    expected = labels.read(BASELINE.parent / "SUMMARY.json")
    assert q == expected["q"]
    assert deaths == expected["deaths"]
    assert baseline_zenchi == expected["zenchi"]
    assert zenchi == labels.agreement(registered / "zenchi/display.npz", games)
    return dict(equivalence=equal, q=q, zenchi_smoothing_on=zenchi,
                zenchi_e36b=baseline_zenchi, deaths=deaths)


def main() -> None:
    """再生済み成果物だけを読み、照合結果を別ファイルへ保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--registered", type=Path, default=REGISTERED)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.registered)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
