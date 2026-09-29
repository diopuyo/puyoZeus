"""既定OFFのR1照合器。原画像履歴だけを保持し通常確定を待たせない。"""
from __future__ import annotations
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable
import cv2
import numpy as np
from src.animation_filter import AnimationFilter
from src.board import BOARD_ROWS, BOARD_COLS, HIDDEN_ROWS, COLOR_OJAMA, COLOR_EMPTY
from src.board_state_machine import BoardState
from src.effect_glow_detector import is_effect_glow_active
from src.image_reader import ColorClassifier, DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.ojama_visual_detector import OJAMA_ROI_HEIGHT
from src.placement_signal_images import next_translation, multiply, NEXT_DEBOUNCE_SEC
from src.placement_signal_reconcile import Observation, reconcile, select_observation, FORMULA_LOOKBACK_FRAMES
from src.score_ocr import FORMULA_MULT_NCC_MIN

FRAME_SIZE = (1920, 1080)
HISTORY_SIZE = FORMULA_LOOKBACK_FRAMES + 2
VISIBLE = frozenset(range(HIDDEN_ROWS, BOARD_ROWS))
ERASURE_GROUP_SIZE = 4


class PlacementSignalRuntime:
    """NEXT周期の最初の合図だけを消費し、採否と全差分を記録する。"""

    def __init__(self, cnn: Any, template: np.ndarray) -> None:
        self.cnn, self.template, self.hsv = cnn, template, ColorClassifier()
        self.audit: list[dict] = []
        self.sink: Callable[[dict], None] | None = None
        self.reset()

    def reset(self) -> None:
        """試合境界では合図・画像窓を破棄し、監査記録は残す。"""
        self.history = (deque(maxlen=HISTORY_SIZE), deque(maxlen=HISTORY_SIZE))
        self.filters = (AnimationFilter(), AnimationFilter())
        self.gray: np.ndarray | None = None
        self.last_frame = -1
        self.last_motion = [-float('inf'), -float('inf')]
        self.formula = [False, False]
        self.consumed = [False, False]

    def record(self, row: dict) -> None:
        """通常処理の原票を変えず、照合の監査を外部へ出す。"""
        if self.sink is not None:
            self.sink(row)
        else:
            self.audit.append(row)

    def log_to(self, path: Path) -> None:
        """中断時にも確認できる追記形式で全採否を保存する。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('', encoding='utf-8')
        def write(row: dict) -> None:
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        self.sink = write

    def read(self, frame: np.ndarray, index: int, stamp: float, pipe: Any) -> list[Observation]:
        """同一パッチをCNN単独・既定HSV単独で読み、既存品質検査を流用する。"""
        regions = (DEFAULT_P1_REGION, DEFAULT_P2_REGION)
        patches = [frame[y1:y2, x1:x2] for region in regions
                   for r in sorted(VISIBLE) for c in range(BOARD_COLS)
                   for x1, y1, x2, y2 in [region.cell_sample_rect(r, c)]]
        if self.cnn is None:
            raise ValueError('CNN単独読取器なし')
        colors = np.asarray(self.cnn.classify_batch(patches)).reshape(2, BOARD_ROWS-HIDDEN_ROWS, BOARD_COLS)
        hsv = np.asarray([self.hsv.classify(p) for p in patches]).reshape(colors.shape)
        observations = []
        for side, region in enumerate(regions):
            grid, hgrid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8), np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.uint8)
            grid[HIDDEN_ROWS:], hgrid[HIDDEN_ROWS:] = colors[side], hsv[side]
            quality = self.filters[side].is_animation(frame, (region.x, region.y, region.width, region.height)).reason
            if is_effect_glow_active(frame, region, VISIBLE):
                quality = quality or 'effect_glow'
            suffix = f'{side+1}p'
            if getattr(pipe, f'_all_clear_pending_{suffix}', False):
                quality = quality or 'all_clear'
            obs = Observation(index, stamp, grid, hgrid, quality)
            observations.append(obs)
        return observations

    def observe(self, pipe: Any, index: int, stamp: float, frame: np.ndarray) -> None:
        """間引き前に呼ぶ。欠落フレームは連続一致として使わない。"""
        if index == self.last_frame:
            return
        if index < self.last_frame:
            self.reset()
        if frame.shape[:2] != (FRAME_SIZE[1], FRAME_SIZE[0]):
            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        try:
            observations = self.read(frame, index, stamp, pipe)
            formula = multiply(frame, self.template)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for side, obs in enumerate(observations):
                obs = self.mark_erasure(side, obs)
                signals = self.signals(side, obs, gray, formula[side])
                self.history[side].append(obs)
                for signal in signals[:1]:
                    self.apply(pipe, side, signal, obs)
            self.gray = gray
        except (ValueError, RuntimeError, cv2.error) as error:
            self.record(dict(frame=index, t_sec=stamp, reason='reader_error', detail=str(error), corrections=[]))
            self.reset()
            # 故障後に別の合図で同じ置きを再照合しない。
            self.consumed = [True, True]
        self.last_frame = index

    def mark_erasure(self, side: int, obs: Observation) -> Observation:
        """一致観測で同色4個以上が消えている画像を消去途中として記録する。"""
        if not self.history[side]:
            return obs
        previous = self.history[side][-1]
        lost = previous.agreement() & obs.agreement() & (obs.cnn == 0)
        erasing = any(np.count_nonzero(lost & (previous.cnn == color)) >= ERASURE_GROUP_SIZE
                      for color in range(1, 6))
        return replace(obs, erasing=erasing)

    def signals(self, side: int, obs: Observation, gray: np.ndarray, formula: float) -> list[str]:
        """NEXT移動・式の立上り・新規おじゃま上端流入を時点通り検出する。"""
        signals = []
        history = self.history[side]
        previous = history[-1] if history else None
        if self.gray is not None and previous is not None and previous.frame == obs.frame-1:
            if next_translation(self.gray, gray, side)['moving']:
                if obs.stamp-self.last_motion[side] > NEXT_DEBOUNCE_SEC:
                    signals.append('next')
                self.last_motion[side] = obs.stamp
        visible = formula >= FORMULA_MULT_NCC_MIN
        if visible and not self.formula[side]:
            signals.append('formula')
        self.formula[side] = visible
        if previous is not None and previous.frame == obs.frame-1:
            new = obs.agreement() & (obs.cnn == COLOR_OJAMA) & (previous.cnn != COLOR_OJAMA)
            if new[HIDDEN_ROWS:HIDDEN_ROWS+OJAMA_ROI_HEIGHT].any():
                signals.append('ojama')
        return signals

    def apply(self, pipe: Any, side: int, signal: str, latest: Observation) -> None:
        """最初の合図を消費し、品質不良も再試行せず記録だけ残す。"""
        consumed = self.consumed[side]
        self.consumed[side] = True
        history = list(self.history[side])
        selected = select_observation(history, signal, latest.frame)
        previous = next((o for o in history if selected and o.frame == selected.frame-1), None)
        suffix = f'{side+1}p'
        context = getattr(pipe, f'_sm_{suffix}').context
        before = context.confirmed_board._grid.tolist() if context.confirmed_board else None
        fixed, audit = reconcile(context.confirmed_board, selected, previous)
        if consumed:
            fixed, audit['reason'] = None, 'already_consumed'
        elif context.state != BoardState.STABLE:
            fixed, audit['reason'] = None, 'nonstable'
        elif getattr(pipe, f'_active_chain_{suffix}', None) is not None:
            fixed, audit['reason'] = None, 'chain_animation'
        if fixed is not None:
            self.commit(pipe, side, fixed, audit['corrections'])
        else:
            audit['corrections'] = []
            for cell in audit['cell_reasons']:
                if cell['reason'] == 'corrected':
                    cell['reason'] = audit['reason']
        self.record(dict(audit, side=side, signal=signal, frame=latest.frame, t_sec=latest.stamp,
                         observed_frame=selected.frame if selected else None,
                         before=before))
        if signal == 'next':
            self.consumed[side] = False

    @staticmethod
    def commit(pipe: Any, side: int, fixed: Any, cells: list[dict]) -> None:
        """既存確定経路を使い、修正セルの古い投票による巻戻しを防ぐ。"""
        suffix, label = f'{side+1}p', f'{side+1}P'
        context = getattr(pipe, f'_sm_{suffix}').context
        pipe._set_deferred_confirmed(label, fixed, context)
        setattr(pipe, f'_prev_confirmed_{suffix}', fixed.copy())
        grace = getattr(pipe, f'_landing_grace_{suffix}', None)
        if grace is not None:
            setattr(pipe, f'_landing_grace_{suffix}', (grace[0], fixed.copy(), grace[2]))
        positions = {(cell['row'], cell['col']) for cell in cells}
        for entry in getattr(pipe, f'_pending_landing_vote_{suffix}', []):
            entry['cells'] = [cell for cell in entry['cells'] if tuple(cell[:2]) not in positions]
        history = getattr(pipe, f'_stable_cnn_history_{suffix}', {})
        for position in positions:
            history.pop(position, None)
        watch = getattr(pipe, f'_landing_color_watch_{suffix}', [])
        setattr(pipe, f'_landing_color_watch_{suffix}', [w for w in watch if w[0] not in positions])
        guard = getattr(pipe, f'_piece_persistence_{suffix}', None)
        if guard is not None:
            for position in positions:
                guard._protected.pop(position, None)
                if fixed.get(*position) != COLOR_EMPTY:
                    guard._protected[position] = int(fixed.get(*position))
        glow = getattr(pipe, f'_glow_guard_{suffix}', None)
        if glow is not None and glow.frozen_board is not None:
            frozen = glow.frozen_board.copy()
            for row, col in positions:
                frozen.set(row, col, int(fixed.get(row, col)))
            glow.frozen_board = frozen
