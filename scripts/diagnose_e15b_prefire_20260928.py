"""E15の発火前低下をフレーム別に計装する。推論ロジックは変更しない。"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
from scripts.replay_exchange_event_20260926 import replay
from src.exchange_event_evaluator import build_features, count_sides
from src.exchange_event_count_features import orient
from src.exchange_event_features import S1_COLUMNS, S3_COLUMNS
from src.exchange_event_tracker import ExchangeEventTracker
from src import exchange_event_landing as landing_module
from src.board import COLOR_OJAMA, DEATH_COL
from src.indicators_v2 import NEAR_FUTURE_K_LEVELS

OUT = Path("logs/e15b/prefire")
RECORD = Path("logs/review_zenchi_g41_43_e14/inputs.jsonl.gz")
MODEL = Path("models/exchange_event_v3")
START, END = 2611.9, 2614.1
COUNT_START, SCORE_START = len(S3_COLUMNS), len(S1_COLUMNS)
NF_COUNT = len(NEAR_FUTURE_K_LEVELS)


class Trace:
    """確定盤面変更・S3実入力・着弾合成値を同じフレームで保存する。"""
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.last_keys: list[Any] = [None, None]
        self.board_changed_at: list[float | None] = [None, None]
        self.queue_changed_at: list[float | None] = [None, None]
        self.last_vector: np.ndarray | None = None
        self.captured: tuple | None = None
        self.counterfactuals: list[dict] = []
        self.landing_m0: float | None = None

    def physical(self, overlay: Any, idx: int, landing: dict) -> dict:
        """仮想着弾評価へ渡った実盤面と仮想盤面の物理量を記録する。"""
        boards = tuple(s.board for s in overlay._landing_projection.latest)
        incoming = landing.get("incoming", (0, 0))[idx]
        virtual, _, _ = landing_module.land_pending_ojama_onto_board(boards[idx], boards[1-idx], incoming)
        grid = boards[idx]._grid
        return dict(board_puyos=int(np.count_nonzero(grid)),
                    board_ojama=int(np.count_nonzero(grid == COLOR_OJAMA)),
                    center_height=boards[idx].height_of(DEATH_COL),
                    virtual_puyos=int(np.count_nonzero(virtual._grid)), virtual_dead=virtual.is_dead())

    def capture(self, tracker: Any, event: Any, source: str, stamp: float) -> None:
        """S3の実入力を複製し、結果を変更せず側別countも保存する。"""
        if START <= stamp <= END and source in ("S3", "S3_provisional"):
            _, vector, side = build_features(event, tracker.models.elapsed_thresholds)
            counts = count_sides(event)
            self.captured = (np.r_[vector, orient(counts, side)], counts, side)

    def ablate(self, tracker: Any, stamp: float) -> None:
        """直前フレームへ特徴群を一つだけ戻す感度を測る。加法寄与ではない。"""
        if self.captured is None:
            return
        vector, counts, side = self.captured
        if self.last_vector is not None and not np.array_equal(vector, self.last_vector, equal_nan=True):
            result = dict(t_sec=stamp, actual_s3=self.predict(tracker, vector, side))
            old = self.last_vector[COUNT_START:]
            old_counts = np.stack((np.r_[old[:NF_COUNT], old[-2]],
                                   np.r_[old[NF_COUNT:2*NF_COUNT], old[-1]]))
            old_counts = old_counts[[side, 1-side]]
            for idx in range(2):
                for name, columns in (("NF", slice(None, -1)), ("margin", -1)):
                    changed = counts.copy()
                    changed[idx, columns] = old_counts[idx, columns]
                    probe = np.r_[vector[:COUNT_START], orient(changed, side)]
                    result[f"restore_{idx+1}P_{name}"] = self.predict(tracker, probe, side)
            probe = vector.copy()
            probe[SCORE_START:COUNT_START] = self.last_vector[SCORE_START:COUNT_START]
            result["restore_scores"] = self.predict(tracker, probe, side)
            self.counterfactuals.append(result)
        self.last_vector = vector.copy()

    @staticmethod
    def predict(tracker: Any, vector: np.ndarray, side: int) -> float:
        """学習済みモデルだけを再評価し、trackerの状態へ書き戻さない。"""
        p = tracker.models.predict_source_probability("S3", vector)
        return p if side == 0 else 1-p

    def __call__(self, overlay: Any, inputs: tuple) -> None:
        """観測済み確定値の変化時刻を追跡し、指定区間だけ全フレーム出力する。"""
        tracker, stamp = overlay.tracker, inputs[3]
        changes = []
        for idx, history in enumerate(overlay._history):
            key = (history[-1].board._grid.tobytes(), history[-1].queue.tobytes()) if history else None
            old = self.last_keys[idx]
            changed = [key is not None and (old is None or old[i] != key[i]) for i in range(2)]
            if changed[0]:
                self.board_changed_at[idx] = stamp
            if changed[1]:
                self.queue_changed_at[idx] = stamp
            self.last_keys[idx] = key
            changes.append(changed)
        if not START <= stamp <= END:
            return
        record = tracker.current
        base = next(v for v in reversed(record.values) if v["source"] in ("S3", "S3_provisional"))
        landing = overlay._landing_projection.last or {}
        row = dict(t_sec=stamp, source=tracker.source, s3=base["p1"], s3_sec=base["t_sec"],
            landing=landing.get("gfe_p1"), landing_sec=landing.get("t_sec"), combined=tracker.probability,
            landing_m0=self.landing_m0)
        for idx in range(2):
            prefix = f"{idx+1}P_"
            counts = tracker.count_sides[idx]
            values = {f"NF_k{k+1}": float(np.expm1(v)) for k, v in enumerate(counts[:-1])}
            values.update(counter_margin=float(np.sign(counts[-1])*np.expm1(abs(counts[-1]))),
                incoming=landing.get("incoming", [None, None])[idx], hands=landing.get("hands", [None, None])[idx],
                board_changed=changes[idx][0], queue_changed=changes[idx][1],
                board_update_sec=self.board_changed_at[idx], next_update_sec=self.queue_changed_at[idx],
                confirmed_sec=overlay._history[idx][-1].t_sec,
                state=(inputs[0].p1, inputs[0].p2)[idx].state.name)
            values.update(self.physical(overlay, idx, landing))
            row.update({prefix+k: v for k, v in values.items()})
        self.rows.append(row)
        self.ablate(tracker, stamp)


def main() -> None:
    """記録再生の実値と条件付き感度を保存し、既存結果との一致を検査する。"""
    trace = Trace()
    original = ExchangeEventTracker._evaluate
    original_landing = landing_module.evaluate_exchange_event
    def evaluate(tracker: Any, event: Any, source: str, stamp: float) -> bool:
        success = original(tracker, event, source, stamp)
        if success:
            trace.capture(tracker, event, source, stamp)
        return success
    def evaluate_landing(event: Any, models: Any) -> float:
        trace.landing_m0 = event.m0_probability_1p
        return original_landing(event, models)
    ExchangeEventTracker._evaluate = evaluate
    landing_module.evaluate_exchange_event = evaluate_landing
    try:
        replay(RECORD, OUT, MODEL, True, trace)
    finally:
        ExchangeEventTracker._evaluate = original
        landing_module.evaluate_exchange_event = original_landing
    with (OUT / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace.rows[0]))
        writer.writeheader()
        writer.writerows(trace.rows)
    (OUT / "sensitivity.json").write_text(json.dumps(trace.counterfactuals, indent=2), encoding="utf-8")
    reference = json.loads(Path("logs/e15/on/review/count_trace.json").read_text())
    probabilities = {r["t_sec"]: r["p1"] for r in reference}
    assert all(r["combined"] == probabilities[r["t_sec"]] for r in trace.rows)
    print(f"元E15と勝率完全一致: {len(trace.rows)}フレーム", flush=True)


if __name__ == "__main__":
    main()
