"""記録設計の試作。認識結果を変更せず、因果的な観測と候補を追記する。"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any

import numpy as np

COLOR_MIN, COLOR_MAX, OJAMA = 1, 5, 9
CHAIN_STATES = ("CHAIN", "GRAVITY_SETTLE")
SCORE_FACTOR = 10


def grid(board: Any) -> list[list[int]] | None:
    """盤面だけを複製し、画像や可変参照を保存しない。"""
    return None if board is None else board._grid.astype(int).tolist()


def count(board: list | None, garbage: bool = False) -> int | None:
    """不明セルを含む盤面は完全な個数として扱わない。"""
    if board is None:
        return None
    cells = np.asarray(board)
    if np.any(~np.isin(cells, (0, 1, 2, 3, 4, 5, OJAMA))):
        return None
    mask = cells == OJAMA if garbage else (cells >= COLOR_MIN) & (cells <= COLOR_MAX)
    return int(mask.sum())


def difference(before: list | None, after: list | None, garbage: bool = False) -> int | None:
    """クリップせず符号付きの増分を残す。"""
    a, b = count(before, garbage), count(after, garbage)
    return None if a is None or b is None else b-a


@dataclass
class SideHistory:
    """イベントを繋ぐ外部状態。指標・認識器へ書き戻さない。"""
    stable: dict | None = None
    active: dict | None = None
    last_state: str = "MENU"
    last_trigger: float | None = None
    sequence: int = 0
    ojama_entry: dict | None = None


@dataclass
class RecordObserver:
    """未来からの訂正は追記し、発火時の記録を上書きしない。"""
    sides: dict[str, SideHistory] = field(default_factory=lambda: {s: SideHistory() for s in ("1P", "2P")})
    last_game: int | None = None
    last_time: float | None = None

    def update(self, row: dict) -> list[dict]:
        """同一フレームを基準に両側を観測する。"""
        if self.last_time is not None and row["t_sec"] <= self.last_time:
            raise ValueError("記録時刻は厳密な単調増加が必要です")
        self.last_time = row["t_sec"]
        events = []
        if self.last_game != row["game_idx"]:
            for name, history in self.sides.items():
                if history.active is not None:
                    events.append(self._event(row, name, "fire_censored", reason="boundary"))
            self.sides = {s: SideHistory() for s in self.sides}
            self.last_game = row["game_idx"]
        for side in row["sides"]:
            events.extend(self._side(row, side))
        return events

    def _event(self, row: dict, side: str, kind: str, **values: Any) -> dict:
        """イベント時刻と利用可能時刻を分離する。"""
        return dict(kind=kind, side=side, frame=row["frame"], game_idx=row["game_idx"],
                    observed_at_sec=row["t_sec"], **deepcopy(values))

    def _start(self, row: dict, side: dict, history: SideHistory) -> dict:
        """起点盤面候補を厳密な発火前観測に限定する。"""
        event = side["chain"]
        if event is not None and event["trigger_sec"] > row["t_sec"]:
            event = None
        trigger = event["trigger_sec"] if event else row["t_sec"]
        pre = history.stable
        if pre is not None and pre["t_sec"] >= trigger:
            pre = None
        history.sequence += 1
        history.active = dict(fire_id=f'{row["game_idx"]}:{side["side"]}:{history.sequence}',
                              trigger_sec=trigger, pre=pre, steps=[],
                              tsumo_at_start=side["tsumo_count"])
        return self._event(row, side["side"], "fire_start", **history.active,
                           chain_before=event["before_board"] if event else None,
                           quality="candidate_not_ground_truth")

    def _finish(self, row: dict, side: dict, history: SideHistory) -> dict:
        """連鎖後に初めて消費量を利用可能にする。未補正差分は代理量。"""
        active = history.active
        pre = active["pre"]
        before = None if pre is None else pre["board"]
        delta = difference(before, side["confirmed_board"])
        steps = active["steps"]
        consumed = None if delta is None else -delta
        erased = sum(s["erased"] for s in steps) if steps else None
        history.active = None
        return self._event(row, side["side"], "fire_complete", **active,
                           post_board=side["confirmed_board"], consumed_board_delta=consumed,
                           consumed_formula=erased, board_formula_equal=None if consumed is None or erased is None else consumed == erased,
                           intervening_tsumo=side["tsumo_count"]-active["tsumo_at_start"],
                           quality="candidate_not_ground_truth")

    def _side(self, row: dict, side: dict) -> list[dict]:
        """発火・式・着弾・設置候補を一方向の観測列として抽出する。"""
        history = self.sides[side["side"]]
        events = []
        chain = side["chain"]
        causal = chain is not None and chain["trigger_sec"] <= row["t_sec"]
        new_trigger = causal and chain["trigger_sec"] != history.last_trigger
        entering = side["state"] in CHAIN_STATES and history.last_state not in CHAIN_STATES
        if (new_trigger or entering) and history.active is None:
            events.append(self._start(row, side, history))
        if causal:
            history.last_trigger = chain["trigger_sec"]
        step = side["formula_step"]
        if step is not None:
            if history.active is not None:
                history.active["steps"].append(step)
            events.append(self._event(row, side["side"], "formula_step", **step))
        if side["state"] == "OJAMA_FALL" and history.ojama_entry is None:
            history.ojama_entry = history.stable
        if side["state"] == "STABLE" and side["confirmed_board"] is not None:
            if history.active is not None:
                events.append(self._finish(row, side, history))
            events.extend(self._settled(row, side, history))
        history.last_state = side["state"]
        return events

    def _settled(self, row: dict, side: dict, history: SideHistory) -> list[dict]:
        """盤面増分を裏付け候補にし、台帳drainを実着弾と同一視しない。"""
        events = []
        old = history.stable
        board = side["confirmed_board"]
        if old is not None and old["board"] != board:
            delta = difference(old["board"], board, garbage=True)
            if delta is not None and delta > 0:
                events.append(self._event(row, side["side"], "landing_candidate",
                    earliest_sec=old["t_sec"], latest_sec=row["t_sec"], net_increase=delta,
                    before_board=old["board"], after_board=board,
                    ojama_phase_seen=history.ojama_entry is not None,
                    ledger_drain_delta=side["ledger_dropped"]-old["ledger_dropped"],
                    quality="net_increment_not_gross_count"))
        if side["landing_diag"] is not None or (old is not None and side["tsumo_count"] > old["tsumo"]):
            events.append(self._event(row, side["side"], "placement_signal",
                tsumo_count=side["tsumo_count"], next_slide=side["next_slide_motion"],
                landing_diag=side["landing_diag"]))
        history.stable = dict(board=board, t_sec=row["t_sec"], frame=row["frame"],
                              tsumo=side["tsumo_count"], ledger_dropped=side["ledger_dropped"])
        history.ojama_entry = None
        return events
