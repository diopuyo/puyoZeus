"""エピソードの観測済み得点と整合する完走候補だけを公開する。"""
from __future__ import annotations
from src.phase_j._live_ex_frozen.src.b3_profile import timed

from dataclasses import dataclass, field

import numpy as np

from src.phase_j._live_ex_frozen.src.board import Board, COLOR_UNKNOWN
from src.phase_j._live_ex_frozen.src.chain import ChainSimulator
from src.phase_j._live_ex_frozen.src.exchange_prefire_candidates import completion, enumerate_candidates, pairs_for, statistics
from src.phase_j._live_ex_frozen.src import puyo_core_bridge as native
from src.phase_j._live_ex_frozen.src.match_color_evidence import GAME_COLOR_COUNT, COLORS
from src.phase_j._live_ex_frozen.src.production_config import GHOST_CHAIN_RULE_ENABLED

NAN = float("nan")


@timed('search')
def origin_candidates(before: np.ndarray | None, colors: tuple[int, ...],
                      queue: tuple[int, ...]) -> tuple[list[dict], str]:
    """発火済み起点を優先し、0連鎖起点のみ本番と同じ2手補完を行う。"""
    if before is None:
        return [], "missing_origin"
    if len(set(np.unique(before)) & COLORS) > GAME_COLOR_COUNT:
        return [], "recognition_conflict"
    if np.any(before == COLOR_UNKNOWN):
        return [], "unknown_origin"
    board = Board.from_list(before.astype(int).tolist())
    simulator = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED, fast_groups=True)
    option = completion(board, simulator)
    if option["prefix"]:
        option["weight"] = 1
        return [option], ""
    if len(colors) != GAME_COLOR_COUNT:
        return [], "unknown_palette"
    options = _completion_candidates(board, colors, queue, simulator)
    return options, "" if options else "no_completion_candidates"


def _completion_candidates(board: Board, colors: tuple[int, ...], queue: tuple[int, ...],
                           simulator: ChainSimulator) -> list[dict]:
    """列挙と段別得点をRustへ融合し、元の候補順と経路の重みを保持する。"""
    raw = native.b3_origin_candidates(board, pairs_for(colors, queue),
                                      simulator._exclude_hidden_row_from_pop)
    if raw is None:
        return enumerate_candidates(board, colors, queue, simulator)[0]
    return raw[0]


@dataclass
class EpisodePrediction:
    """断片の再発火で初期化せず、式の段の累積得点を順に確定する。"""
    options: list[dict] = field(default_factory=list)
    reason: str = "missing_origin"
    available_at: float = NAN
    observed_score: float = 0.0
    observed_count: int = 0
    length: float = NAN
    score: float = NAN
    final: bytes | None = None

    def observe(self, product: float, available_at: float) -> None:
        """段数不足・段別得点不一致の候補は永久に除外する。"""
        self.observed_count += 1
        self.observed_score += product
        count = self.observed_count
        had_options = bool(self.options)
        self.options = [v for v in self.options if len(v["prefix"]) >= count
                        and v["prefix"][count - 1] == self.observed_score]
        if had_options and not self.options:
            self.reason = "observation_conflict"
        self.publish(available_at)

    def publish(self, available_at: float) -> None:
        """A1bの最大段数候補から段数・得点・終端を一緒に選ぶ。"""
        self.available_at = available_at
        self.length = self.score = NAN
        self.final = None
        if not self.options:
            return
        selected = statistics(self.options, elapsed=0.0, max_chain=True)
        self.length = float(selected["predicted_count"])
        # 観測済み得点は確定値。残りだけ同じ候補の予測を足す。
        remaining = float(selected["selected_score"]) - self.observed_score
        self.score = self.observed_score + remaining
        self.final = np.asarray(selected["board"], dtype=np.int8).tobytes()
        self.reason = ""
