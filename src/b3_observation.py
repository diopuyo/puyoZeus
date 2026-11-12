"""B2記録と同じ観測を、本番認識の返り値と当該画像から作る。"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from scripts.collect_boards_lean import _SideState, _drive_ojama_accounting_lean
from src.board_state_machine import BoardState
from src.ojama_accounting import OjamaAccountingTracker
from src.ojama_warning import OjamaWarningDetector
from src.record_redesign_observer import RecordObserver, grid, SCORE_FACTOR
from src.score_ocr import FormulaStepAccumulator, ScoreOcr
from src.phase_j.live_stage_timing import measure
from src.production_config import (B3_SCORE_MATMUL_DEFAULT, B3_WARNING_PARALLEL_DEFAULT,
                                   B3_WARNING_WORKERS, B3_SKIP_ICON_READING_DEFAULT,
                                   B3_WARNING_BATCH_DEFAULT, B3_WARNING_PREFETCH_DEFAULT,
                                   B3_WARNING_ASYNC_DEFAULT)

ZERO_CONFIRM = 2
CHAIN_FIELDS = ('trigger_sec', 'end_sec', 'chain_count', 'total_erased',
                'total_score', 'mechanism', 'score_estimated')


class ObservationCapture:
    """B2 Pilotの境界・式・イベント契約を保ち、認識本体は再実行しない。"""

    def __init__(self, enable_score_matmul: bool = B3_SCORE_MATMUL_DEFAULT,
                 enable_warning_parallel: bool = B3_WARNING_PARALLEL_DEFAULT,
                 warning_workers: int = B3_WARNING_WORKERS,
                 skip_icon_reading: bool = B3_SKIP_ICON_READING_DEFAULT,
                 enable_warning_batch: bool = B3_WARNING_BATCH_DEFAULT,
                 warning_prefetch: bool = B3_WARNING_PREFETCH_DEFAULT,
                 enable_warning_async: bool = B3_WARNING_ASYNC_DEFAULT) -> None:
        self.ocr = (ScoreOcr.load_default(enable_matmul_ncc=True) if enable_score_matmul
                    else ScoreOcr.load_default())
        if skip_icon_reading:
            self.warning = None
        elif enable_warning_batch or enable_warning_async:
            # 試作 (2026-10-23、既定 OFF): CNN を 1 コマ 1 回の forward に。先読みは盤面認識と重ねる。
            from src.batched_ojama_warning import BatchedOjamaWarningDetector, PrefetchingWarning
            detector = BatchedOjamaWarningDetector()
            self.warning = (PrefetchingWarning(detector)
                            if warning_prefetch and not enable_warning_async else detector)
        elif enable_warning_parallel:
            from src.parallel_ojama_warning import ParallelOjamaWarningDetector
            self.warning = ParallelOjamaWarningDetector(warning_workers)
        else:
            self.warning = OjamaWarningDetector()
        if self.warning is not None and (self.warning._cnn is None or not self.warning._templates):
            raise FileNotFoundError('B3観測用のojama_cnn.pt / ui_templates/ojamaが必要です')
        from scripts.b2_recollect_cache import enable
        if enable_warning_parallel:
            enable(self, thread_safe=True)
        else:
            enable(self)
        self.account = OjamaAccountingTracker()
        self.account.reset()
        self.states = (_SideState(), _SideState())
        self.previous = (BoardState.MENU, BoardState.MENU)
        self.accum = {s: FormulaStepAccumulator() for s in ('1P', '2P')}
        self.prev_count = {s: None for s in ('1P', '2P')}   # 依頼 §4: 置き (tsumo_count の増え) を式の読みに渡す
        self.observer = RecordObserver()
        self.zero_run, self.game = 0, 0
        self.zero_latched = self.seen_positive = False
        self._warning_async = None
        if enable_warning_async and self.warning is not None:
            from src.async_ojama_warning import AsyncWarning
            self._warning_async = AsyncWarning(self.warning)

    def close(self) -> None:
        """予告読取が所有する worker を終了する。"""
        if getattr(self, '_warning_async', None) is not None:
            self._warning_async.close()
        close = getattr(self.warning, 'close', None)
        if close is not None:
            close()

    def prefetch_warning(self, image: Any) -> bool:
        """先読みつきの予告読取 (b3_warning_prefetch) のときだけ、同じコマの読取を先に始める。"""
        prefetch = getattr(self.warning, 'prefetch', None)
        return bool(prefetch(image)) if prefetch is not None else False

    def boundary(self, scores: Any) -> None:
        values = (scores.score_1p, scores.score_2p)
        self.zero_run = self.zero_run + 1 if values == (0, 0) else 0
        if self.seen_positive and self.zero_run >= ZERO_CONFIRM and not self.zero_latched:
            self.game += 1
            self.zero_latched, self.seen_positive = True, False
            for accum in self.accum.values():
                accum.reset()
        if any(v is not None and v > 0 for v in values):
            self.seen_positive, self.zero_latched = True, False

    def side(self, side: Any, frame: Any, t: float, snap: Any,
             scores: Any, count: int) -> dict:
        name = side.side
        raw = measure('b3_formula', self.ocr.read_formula_side, frame, name)
        displayed = scores.score_1p if name == '1P' else scores.score_2p
        accum = self.accum[name]
        prev, self.prev_count[name] = self.prev_count[name], count
        placed = prev is not None and count is not None and count > prev
        step = accum.update(t, raw, score_displayed=displayed is not None, placed=placed)
        record = None if step is None else dict(asdict(step), erased=step.left // SCORE_FACTOR,
            multiplier=step.right, product=step.product, observed_ordinal=accum.step_count,
            ordinal_quality='observed_order_not_verified_chain_number')
        chain = side.chain_event
        chain = None if chain is None else dict({k: getattr(chain, k) for k in CHAIN_FIELDS},
                                               before_board=grid(chain.before_board))
        key = 'p1' if name == '1P' else 'p2'
        result = dict(side=name, state=side.state.name, confirmed_board=grid(side.confirmed_board),
            raw_cnn_board=grid(side.raw_cnn_board), board_provenance=side.board_provenance,
            score=side.score, chain=chain, formula=asdict(raw), formula_step=record,
            tsumo_count=count, next_slide_motion=side.next_slide_motion, landing_diag=side.landing_diag,
            ledger_forecast=getattr(snap, f'forecast_{key}'),
            ledger_dropped=getattr(snap, f'total_dropped_to_{key}'))
        from src.chain_continuity import end_enabled
        if end_enabled():
            result.update(next_pair=side.next_pair, dnext_pair=side.dnext_pair)
        from src.light_consumption import enabled
        if enabled() and getattr(side, 'light_cell_probs', None) is not None:
            result['light_cell_probs'] = side.light_cell_probs
        return result

    def update(self, result: Any, image: Any, fi: int, t: float,
               counts: tuple[int, int], scores: Any = None) -> tuple[dict, tuple[dict, ...]]:
        scores = measure('b3_score', self.ocr.read, image) if scores is None else scores
        self.boundary(scores)
        snap = measure('b3_ledger', _drive_ojama_accounting_lean, self.account, *self.states, *self.previous,
                                           result.p1, result.p2, *counts, t)
        self.previous = (result.p1.state, result.p2.state)
        metadata = None
        if getattr(self, '_warning_async', None) is not None:
            warnings, metadata = measure('b3_warning', self._warning_async.update, image, fi, t)
        else:
            warnings = (None, None) if self.warning is None else measure('b3_warning', self.warning.detect, image)
        sides = [self.side(s, image, t, snap, scores, count)
                 for s, count in zip((result.p1, result.p2), counts)]
        from src.light_consumption import annotate_sides
        sides = annotate_sides(self, sides, (result.p1.next_pair, result.p2.next_pair), t, self.game)
        for side, warning in zip(sides, warnings):
            side['icon'] = (None if warning is None else
                           dict(asdict(warning), quality='detector_estimate_no_confidence_api'))
        row = dict(kind='frame', frame=fi, t_sec=t, available_at_sec=t, game_idx=self.game,
                   match_active=result.is_match_active, match_end_locked=result.match_end_locked,
                   sides=sides, ledger=asdict(snap))
        if metadata is not None:
            row['warning_async'] = metadata
        return row, tuple(measure('b3_observer', self.observer.update, row))
