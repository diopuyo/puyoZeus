"""E10cの固定基準とE10bからの全指標差を、元の分母で照合する。"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from scripts import report_e10b_exchange_20260927 as previous
from scripts.aggregate_e3_exchange_eval_20260926 import outcomes, FPS
from scripts.report_e9_exchange_20260927 import metric_rows
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

OUT = Path("logs/e10c")
LOSS_TOLERANCE = .005
AGREEMENT_MIN = .7399
CAUSES = {
    4: ("f+a", "連鎖中の凍結盤面がrow1占有。受け側3連鎖の通知欠落"),
    16: ("b", "表示得点差44100点を再攻撃化。受け側相殺の参加登録欠落"),
    24: ("f+a+b", "連鎖中の凍結row1占有と、未相殺の99015点を使用"),
    29: ("a+b", "消去前9段・消去後7段。22個の端数配置で死を断定"),
    30: ("f+a", "天井設置から消去への移行と4連鎖を遅延認識"),
    49: ("a+b", "受け側連鎖の通知・相殺欠落。31749点を未着弾扱い"),
}


def deaths() -> list[dict]:
    """E10bの公式ラベル・補完ラベルを固定して全発火を照合する。"""
    games = previous.read(previous.ZENCHI / "official_games.json")
    rows = previous.death_rows(OUT / "zenchi", "zenchi", games)
    panel = previous.read(Path("logs/e10b/panel_outcomes.json"))
    for source in SOURCES:
        windows, _ = outcomes(source)
        seconds = [dict(start=w["start"]/FPS, end=w["end"]/FPS, winner=w["winner"]) for w in windows]
        rows.extend(previous.death_rows(OUT / "renders" / source / "on", source, seconds))
    for row in rows:
        if row["winner"] is not None:
            continue
        label = next((r for r in panel if r["source"] == row["source"]
                      and r["game_idx"] == row["game_idx"]), None)
        if label is not None and label["winner"] is not None:
            row.update(winner=label["winner"], false_positive=label["winner"] == row["side"],
                       provenance="panel_outcomes.json")
    return rows


def classification() -> list[dict]:
    """修正前6件の分類と確認した2枚の実画面パスを保存する。"""
    rows = []
    for case in previous.read(Path("logs/e10b/report.json"))["deaths"]:
        if not case["false_positive"]:
            continue
        category, reason = CAUSES[case["exchange_id"]]
        frames = [str(OUT / "frames" / f"case_{case['exchange_id']}_{case['first_sec']+offset:.3f}.jpg")
                  for offset in (0, 2)]
        rows.append(dict(source=case["source"], exchange_id=case["exchange_id"],
            t_sec=case["first_sec"], category=category, reason=reason, frames=frames))
    return rows


def main() -> None:
    """未達基準もそのまま保存し、合否を上書きしない。"""
    rows = deaths()
    summary = dict(firings=len(rows), false_positives=sum(r["false_positive"] is True for r in rows),
                   unlabelled=sum(r["winner"] is None for r in rows))
    summary["false_fraction"] = summary["false_positives"] / max(1, len(rows)-summary["unlabelled"])
    games = previous.read(previous.ZENCHI / "official_games.json")
    agreement = {v: previous.agreement(Path(f"logs/{v}/zenchi/display.npz"), games)
                 for v in ("e10", "e10b", "e10c")}
    pooled = {v: previous.read(Path(f"logs/{v}/v2/pooled.json")) for v in ("e10", "e10b", "e10c")}
    metrics = [dict(metric=a, E10b=b, E10c=c) for (a, b), (_, c) in
               zip(metric_rows(pooled["e10b"]), metric_rows(pooled["e10c"]))]
    previous.OUT = OUT
    display = np.load(OUT / "zenchi/display.npz")
    gates = previous.hold_checks(display)
    gates.update(deaths=summary["unlabelled"] == 0 and summary["false_fraction"] <= .05,
        zenchi=agreement["e10c"]["agreement"] >= AGREEMENT_MIN,
        q_log_loss=pooled["e10c"]["M3_q"]["on"]["groups"]["all"]["log_loss"] <=
            pooled["e10"]["M3_q"]["on"]["groups"]["all"]["log_loss"] + LOSS_TOLERANCE)
    gates["off_sha"] = previous.read(OUT / "off_sha256.json")["identical"]
    gates["related_tests"] = (OUT / "related_tests.exit").read_text().strip() == "0"
    report = dict(gates=gates, death_summary=summary, deaths=rows, zenchi=agreement,
                  metrics=metrics, classification=classification())
    save_json(OUT / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "deaths"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zenchi-only", action="store_true")
    if parser.parse_args().zenchi_only:
        previous.OUT = OUT
        print(previous.hold_checks(np.load(OUT / "zenchi/display.npz")))
        print(previous.agreement(OUT / "zenchi/display.npz",
                                 previous.read(previous.ZENCHI / "official_games.json")))
    else:
        main()
