"""学習収集器の採用条件を再利用し、採用したNEXT行だけを通知する。"""
from __future__ import annotations

import shlex
from typing import Any

import numpy as np

from scripts import collect_boards_lean as lean
from src.board_state_machine import BoardState
from src.chain import ChainSimulator
from src.production_config import GHOST_CHAIN_RULE_ENABLED, collect_flags
from src.b3_realtime import QueueObservation

SIDE_NAMES = ('1P', '2P')


class LiveQueues:
    """NPZの代わりに当該フレーム分だけ保持するappend先。"""

    def __init__(self) -> None:
        self.flags = set(shlex.split(collect_flags()))
        self.states = (lean._SideState(), lean._SideState())
        self.shared = lean._SharedGameCounter(
            multisignal_mode=self.enabled('boundary_multisignal'),
            require_newmatch_evidence=self.enabled('boundary_newmatch_evidence'))
        self.physics = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
        self.index = 0
        self.rows: list[QueueObservation] = []
        self.current_game: int | None = None
        # コマ落ちに強い数え方 (2026-10-23、B07 型 H2): 直前に観測したコマ番号。刻みモードでだけ使う。
        self.prev_frame: int | None = None

    def enabled(self, name: str) -> bool:
        return '--enable-' + name.replace('_', '-') in self.flags

    def move_window_frame(self, pipe: Any, frame: int) -> int | None:
        """B07 (型 H2): 1手区切り窓の締切判定に使うコマ番号。OFF (刻みの幅なし) なら None。

        この観測で初めて見えた STABLE の新しい盤面は、前回の観測の直後 (前回 + 1 刻み) に
        起きた可能性があるとみなす (寛容側)。全コマでは「前回 + 1 刻み」= 今のコマなので同じ。
        """
        step = getattr(pipe, 'media_tick_step', None)
        if step is None:
            return None
        previous, self.prev_frame = self.prev_frame, int(frame)
        if previous is None:
            return int(frame)
        window = min(int(frame), previous+int(step))
        from src.media_tick import audit_note
        audit_note('B07', window < int(frame))   # 母数と、締切を前回 + 1 刻みで見た件数 (診断のみ)
        return window

    def append(self, grid: Any, video: str, side: str, t: float, game: int,
               frame: int, **kwargs: Any) -> None:
        queue = tuple(kwargs.get('next_pair') or (-1, -1)) + tuple(kwargs.get('dnext_pair') or (-1, -1))
        self.rows.append(QueueObservation(self.index, side, frame, float(np.float32(t)), queue, game))
        self.current_game = game
        self.index += 1

    def update(self, pipe: Any, result: Any, image: Any, frame: int, t: float,
               counts: tuple[int, int]) -> tuple[QueueObservation, ...]:
        self.rows.clear()
        window_frame = self.move_window_frame(pipe, frame)
        evidence = lean._compute_newmatch_evidence(result) if self.enabled('boundary_newmatch_evidence') else None
        self.shared.observe_visual_signal(result.is_match_active, t, new_match_evidence=evidence)
        for state, side, count in zip(self.states, (result.p1, result.p2), counts):
            self._side(pipe, state, side, image, frame, t, count, window_frame)
        # 勝敗教師と監査履歴はlive特徴に不要。現在の境界状態だけ残す。
        self.shared.advance_times.clear()
        self.shared.anomalies.clear()
        self.shared.rejected_rise_times.clear()
        for state in self.states:
            state.final_scores.clear()
        return tuple(self.rows)

    def _side(self, pipe: Any, state: Any, side: Any, image: Any,
              frame: int, t: float, count: int, window_frame: int | None = None) -> None:
        raw_stable = lean._update_raw_pixel_stable(state, image, side.side, t,
                                                  self.enabled('stable_persistence_gate'))
        flags = {f'enable_{name}': self.enabled(name) for name in (
            'chain_estimate_recording', 'move_segmented_recording', 'physics_persistence_filter',
            'record_time_observation_fix', 'record_time_observation_fix_raw')}
        if window_frame is not None:
            # 刻みモードのときだけ渡す。OFF の呼び出しは引数まで従来と同じ。
            flags['move_window_frame_idx'] = window_frame
        lean._process_side_lean(self, state, side.side, side.confirmed_board, side.state,
            side.score, 'live', t, frame, next_pair=side.next_pair, dnext_pair=side.dnext_pair,
            chain_event=side.chain_event, shared_game=self.shared, exclude_phantom=True,
            tsumo_count=count, raw_pixel_stable=raw_stable, physics_sim=self.physics,
            score_reset_requires_zero=self.enabled('score_reset_requires_zero'),
            estimated_board=side.estimated_board, board_provenance=side.board_provenance,
            cnn_board=side.cnn_board, raw_cnn_board=side.raw_cnn_board,
            frame_bgr=image, image_reader=pipe._reader,
            landing_chain_started=bool(getattr(side, 'landing_chain_started', False)), **flags)
