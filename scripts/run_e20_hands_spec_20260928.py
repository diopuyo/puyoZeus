"""E20: 事前登録した候補基準で仕様手数＋確定的応手を再評価する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts import run_e17_ablation_20260928 as prior
from scripts.run_e18_counter_response_20260928 import halves
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.replay_exchange_event_20260926 import compare

OUT = Path("logs/e20")
OPTIONS = dict(death_guard=True, confirmed_death_hold=True,
               landing_counter_response=True, landing_hands_spec=True)
LOSS_MAX, AGREEMENT_MIN, SCENE_MIN = .5131, .8235, .2703


class HandsTrace(prior.AuditTrace):
    """E19の判断時刻を固定し、オンライン手数と後続の実設置を対応付ける。"""

    def __init__(self, source: str) -> None:
        super().__init__(source)
        self.targets = [g for g in json.loads(Path("logs/e19/CLASSIFICATION.json").read_text())["groups"]
            if g["direction"] == "正→誤" and g["evidence"]["category"] == "可能予測だが着弾前の新発火なし"]
        self.budgets: list[dict] = []
        self.placements: set[tuple] = set()

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        super().__call__(overlay, inputs)
        if self.source != "zenchi":
            return
        projection = overlay._landing_projection
        state = projection.hands_observation
        if state is None:
            return
        for idx, stamps in enumerate(state.placements):
            self.placements.update((overlay._game, idx, t) for t in stamps)
        stamp = inputs[3]
        for group in self.targets:
            evidence = group["evidence"]
            if evidence["decision"]["t_sec"] != stamp:
                continue
            idx = int(group["receiver"] == "2P")
            self.budgets.append(dict(game=group["game_idx"], exchange=group["exchange_id"],
                representative_sec=group["representative"]["t_sec"], decision_sec=stamp,
                receiver=group["receiver"], old_hands=evidence["decision"]["hands"][idx],
                landing_sec=evidence["landing_sec"], e19_time_proxy=evidence["observed_time_budget_hands"],
                **projection._spec_budget(overlay.tracker, 1-idx, stamp)))

    def save_hands(self) -> None:
        """時間の手数換算を実手数と呼ばず、着弾前の観測設置を数える。"""
        if self.source != "zenchi":
            return
        assert len(self.budgets) == len(self.targets) == 8
        for row in self.budgets:
            idx = int(row["receiver"] == "2P")
            row["actual_placement_sec"] = sorted(t for game, side, t in self.placements
                if game == row["game"] and side == idx and row["decision_sec"] < t < row["landing_sec"])
            row["recorded_placements"] = len(row["actual_placement_sec"])
        save_json(OUT/"HANDS_AUDIT.json", self.budgets)


def worker(variant: str, source: str) -> None:
    """固定入力を再生し、基準候補のOFF互換性も独立に確認する。"""
    prior.OUT = OUT
    options = OPTIONS if variant == "on" else dict(death_guard=True, confirmed_death_hold=True)
    trace = HandsTrace(source)
    prior.AuditTrace = lambda _: trace
    prior.locked_worker(variant, source, options)
    trace.save_hands()
    if variant == "off":
        save_json(OUT/"OFF_EQUIVALENCE.json", compare(Path("logs/e19/hold/review"), OUT/"off/review"))


def launch(source: str) -> None:
    """同時実行数を抑え、完了した固定入力単位で再開する。"""
    dest = prior.directory("on", source)
    if (dest/"DONE.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/"runner.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "scripts.run_e20_hands_spec_20260928",
            "--source", source], stdout=stream, stderr=subprocess.STDOUT, check=True)


def report() -> dict:
    """固定母数・固定しきい値だけで採否を決める。"""
    prior.OUT, prior.LOSS_MAX, prior.AGREEMENT_MIN = OUT, LOSS_MAX, AGREEMENT_MIN
    result = prior.report("on")
    scene = result["scenes"]
    scene.update(halves(prior.directory("on", "review")/"count_trace.json"))
    result["gates"].update(scene_mean=scene["prefire_mean"] > SCENE_MIN,
        scene_rise=scene["first_half_mean"] < scene["last_half_mean"])
    result["candidate"] = all(result["gates"].values())
    assert result["zenchi"]["frames"] == 8333
    save_json(OUT/"on/METRICS.json", result)
    return result


def main() -> None:
    """評価開始前に合否と観測仕様を保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=prior.ALL_SOURCES)
    parser.add_argument("--variant", choices=("on", "off"), default="on")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    prior.OUT = OUT
    if args.source:
        worker(args.variant, args.source)
        return
    if not args.report_only:
        save_json(OUT/"PROTOCOL.json", dict(options=OPTIONS, q_max=LOSS_MAX,
            zenchi_min=AGREEMENT_MIN, max_false=1, death_denominator=28,
            scene_min=SCENE_MIN, scene_rise="前半31行 < 後半31行", prefire=prior.PREFIRE,
            recent_intervals=5, animation="過去の同連鎖数の発火→NEXT立上り中央値",
            next_confirmation="NEXT2→NEXT1の繰り上がりを2観測確認。生ROI移動だけでは確定しない",
            missing="未観測時は保証できる1手だけ。固定秒数で補完しない",
            baseline="E15+②+死亡確定表示 / ba614f7"))
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(launch, prior.ALL_SOURCES))
    print(json.dumps(report(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
