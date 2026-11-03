"""24hは起動時に選んだ1モデルを使い続ける。EX欠測も同じモデルで推論する。"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from .live_24h_config import ENV_MODEL, ama_enabled, ex_settings
from .live_24h_model import LEARNING_COLUMNS, RETR_COLUMNS, SECOND_COLUMNS, XC_COLUMNS, load_model
from .live_ex_delay import DelayedAsyncEx
from .live_ex_hold import HOLD_SEC
from .live_xc_fill_frac import XC_FILL_FRAC_COLUMNS, fill_fractions

MS = 1000.0


class Live24h:
    def __init__(self, output: Path, model: Any = None, settings: dict | None = None,
                 worker_factory: Callable = DelayedAsyncEx, clock: Callable = perf_counter,
                 use_ama: bool | None = None) -> None:
        self.ama = ama_enabled() if use_ama is None else use_ama
        self.clock, self.worker = clock, None
        self.model = model
        self.ex_columns = LEARNING_COLUMNS
        self.fill_frac_columns: tuple[str, ...] = ()
        self.epoch: Any = None
        self.game: int | None = None
        self.base_result: dict | None = None
        self.signature: Any = None
        self.revision = 0
        self.last = dict(model_source='C2+EX' if self.ama else 'C2',
                         missing_reason='not_completed' if self.ama else 'ama_off')
        self.audit = (output / 'live_24h.jsonl').open('x', encoding='utf-8')
        try:
            if self.ama:
                self.model = model if model is not None else load_model(Path(os.environ[ENV_MODEL]))
                columns = {c for part in ('static', 'exchange')
                           for c in getattr(getattr(self.model, part, None), 'columns', ())}
                self.fill_frac_columns = tuple(c for c in XC_FILL_FRAC_COLUMNS if c in columns)
                if self.fill_frac_columns:
                    columns.update(XC_COLUMNS)
                self.ex_columns += tuple(c for c in (*RETR_COLUMNS, *SECOND_COLUMNS, *XC_COLUMNS) if c in columns)
                settings = settings if settings is not None else ex_settings()
                self.worker = worker_factory(settings)
            self.settings = dict(settings or {})
            self.write(dict(kind='settings', ama=self.ama, settings=settings,
                            model=os.environ.get(ENV_MODEL) if self.ama else None))
        except Exception:
            self.audit.close()
            raise

    def update(self, frame: dict, epoch: int, captured_at: float) -> None:
        if epoch != self.epoch or frame['game_idx'] != self.game:
            self.base_result, self.signature = None, None
        self.epoch, self.game = epoch, frame['game_idx']
        if self.worker is not None:
            if self.settings.get('row_time', False) or self.settings.get('xc_same_time', False):
                from src.b3_eval_adapter import MAX_HOLD_SEC
                oldest = frame['t_sec'] - MAX_HOLD_SEC
                if self.base_result is not None:
                    oldest = min(oldest, self.base_result['t_sec'])
                self.worker.protect_rows(oldest, epoch)
            self.worker.poll()
            self.flush_events()
            if '_ex' not in frame:
                raise ValueError('24h ama ONのEX観測がありません')
            self.worker.submit(frame['_ex'], epoch, captured_at)

    def selection(self, result: dict) -> tuple[dict, Any, str]:
        select = getattr(self.worker, 'held_available', self.worker.available)
        selected_at = self.clock()
        extra, reason = select(selected_at, result['t_sec'],
                               admitted_at=result.get('captured_at'))
        values = {name: math.nan for name in self.ex_columns}
        if extra is not None:
            values = {name: extra['extra'].get(name, math.nan) for name in self.ex_columns}
        signature = None if extra is None else (extra['epoch'], extra['serial'])
        xc_audit = self.select_xc(values, result['t_sec'], extra)
        if xc_audit.get('xc_source') == 'same_time':
            signature = (signature, 'xc', xc_audit['xc_serial'])
        self.selection_audit = dict(selected_at=self.clock(),
                                    hold={} if extra is None else extra.get('hold_audit', {}),
                                    availability=getattr(self.worker, 'availability_audit', {}))
        if getattr(self, 'settings', {}).get('xc_same_time', False):
            self.selection_audit.update(xc_audit)
        if os.environ.get('PUYO_LIVE_EX_AUDIT') == '1':
            self.write(dict(kind='ex_selection', frame=result['frame'], t=result['t_sec'],
                            selected_at=selected_at, reason=reason, **xc_audit,
                            **self.worker.selection_record(result['t_sec'], extra)))
        return values, signature, reason

    def select_xc(self, values: dict, t: float, extra: dict | None) -> dict:
        from .live_ex_xc_same_time import select_xc
        history = getattr(self.worker, 'xc_history', None)
        key = history.key_at(t) if getattr(self, 'settings', {}).get('xc_same_time', False) and history else {}
        return select_xc(values, key, extra)

    def infer(self, row: dict, values: dict) -> float:
        # 推論中に xc が失効した場合も、欠測にした元の列から作り直す。
        values.update(fill_fractions(values, self.fill_frac_columns))
        probability = self.model.predict_row(dict(row, **values))
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('24hモデルの予測が有限な確率ではありません')
        return float(probability)

    def predict(self, result: dict) -> float:
        if not self.ama:
            return result['raw_probability']
        self.base_result = result
        started = self.clock()
        values, signature, reason = self.selection(result)
        probability = self.infer(result['row'], values)
        _, current, after_reason = self.selection(result)
        if current != signature:
            # xcだけの変更ならEX・retRは保持し、それ以外は従来どおり全欠測。
            from .live_ex_xc_same_time import ex_signature
            xc_only = ex_signature(current) == ex_signature(signature)
            missing = XC_COLUMNS if xc_only else self.ex_columns
            values.update({name: math.nan for name in missing if name in values})
            probability = self.infer(result['row'], values)
            signature = ex_signature(signature) if xc_only else None
            reason = 'changed_during_inference:' + after_reason
            if self.settings.get('xc_same_time', False):
                self.selection_audit.update(self.select_xc_missing(values))
        self.signature = signature
        self.revision += 1
        self.last = dict(model_source='C2+EX', missing_reason=reason,
                         ex_request=signature, ex_inference_ms=(self.clock()-started)*MS)
        self.write(dict(kind='prediction', frame=result['frame'], t_sec=result['t_sec'],
                        raw_probability=probability, learning=values,
                        **self.selection_audit, **self.last))
        self.flush_events()
        return probability

    def select_xc_missing(self, values: dict) -> dict:
        from .live_ex_xc_same_time import select_xc
        return select_xc(values, {}, None)

    def revalidate(self, display: Any) -> None:
        """行を延命せずEX失効を再検査する。行時刻より後の完成を遡って足さない。"""
        result = self.base_result
        if not self.ama or result is None or result['epoch'] != self.epoch or display.failed:
            return
        _, current, _ = self.selection(result)
        if current == self.signature:
            return
        try:
            display.probability = self.predict(result)
        except Exception as exc:
            display.failed = True
            self.write(dict(kind='error', error=repr(exc)))

    def decorate(self, row: dict) -> None:
        row['live24h'] = dict(ama=self.ama, revision=self.revision, **self.last)
        row['b3_model_label'] = '撃ち合い+B3' if self.ama else '実戦勝率: B3 (C2)'
        if 'judgment_id' in row:
            row['judgment_id'] = (*row['judgment_id'], self.revision)

    def record_config(self, value: dict) -> None:
        """実際に推論するモデルと、特徴だけを作る旧モデルを記録で区別する。"""
        value['live24h'] = dict(ama=self.ama, model_source=self.last['model_source'],
                                logical_delay_ms=self.settings.get('logical_delay_ms'),
                                work_limit=self.settings.get('work_limit'), hold_sec=HOLD_SEC)
        if self.ama:
            from .live_obs_trial import C2_FEATURE_COLUMNS
            from src.b3_source import sha256_file
            from src.b3_ensemble import ensemble_record
            path = Path(os.environ[ENV_MODEL])
            value.update(feature_model=value['model'], feature_model_sha256=value['model_sha256'],
                         model=str(path), model_sha256=sha256_file(path), trial_model='C2+EX',
                         feature_columns=C2_FEATURE_COLUMNS + len(self.ex_columns), static_columns=len(self.model.static.columns),
                         exchange_columns=len(self.model.exchange.columns), **ensemble_record(self.model))

    def write(self, value: dict) -> None:
        self.audit.write(json.dumps(value, ensure_ascii=False, default=str) + '\n')
        self.audit.flush()

    def flush_events(self) -> None:
        if self.worker is None:
            return
        for event in self.worker.take_events():
            if event['kind'] == 'result':
                value = dict(event['value'])
                detail = value.pop('result', None)
                if detail:
                    value.update(compute_ms=detail.get('compute_ms'), failure=detail.get('failure'),
                                 work=(detail.get('output') or {}).get('work_units'))
                    value.update({k: detail.get(k) for k in (
                        'retr_why', 'retr_ms', 'retr_diag', 'retr_attack',
                        'xc_why', 'xc_diag', 'compute_ms_by_stage', 'retr_ama')})
                    if getattr(self, 'settings', {}).get('xc_same_time', False):
                        value['xc'] = detail.get('xc')
                event = dict(kind='result', value=value)
            self.write(event)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
            self.flush_events()
        self.audit.close()
