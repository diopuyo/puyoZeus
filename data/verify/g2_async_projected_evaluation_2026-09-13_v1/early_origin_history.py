"""原Jの資格付き履歴だけ早期保存し、Wを解放してから後発consumerへ一回移管する。"""
from contextlib import ExitStack
import hashlib
import json
from typing import Any

FPS = 60
STRIDE = 2
FIELDS = ('kind', 'status', 'exception', 'side', 'frame_idx', 'time_sec', 'token',
          'source_id', 'run_id', 'pipe_object_id', 'software_reset', 'generation', 'generation_after')
HISTORY_NAME = 'EARLY_ORIGIN_HISTORY.jsonl'
CLOSE_NAME = 'EARLY_ORIGIN_HISTORY_CLOSE.json'


def encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(',', ':'))


def compact(row: dict) -> dict:
    value = {key: row[key] for key in FIELDS}
    value['events'] = [dict(stage=item['stage'], active_origin=item.get('active_origin'))
                       for item in row['events'] if item.get('active_origin') is not None]
    value['returned'] = row.get('returned')
    return dict(row=value, original_row_sha256=hashlib.sha256(encoded(row).encode()).hexdigest())


class EarlyHistory:
    def __init__(self, stack: Any, journal: Any, frames: tuple[int, ...],
                 output: Any, witness: Any, reader: Any, *, probability_factory: Any = None) -> None:
        if (not frames or any(type(frame) is not int or frame < 0 for frame in frames)
                or tuple(range(frames[0], frames[-1] + STRIDE, STRIDE)) != frames):
            raise ValueError('early_history_frames')
        self.journal, self.frames, self.output, self.reader = journal, frames, output, reader
        self.pipe = self.stream = self.witness = None
        self.scope = ExitStack()
        self.records: list[str] = []
        self.probability: Any = None
        self.probability_records: list[str] | tuple[str, ...] = []
        self.probability_digest = hashlib.sha256()
        self.digest = hashlib.sha256()
        self.count, self.last_frame, self.last_steps = 0, -1, -1
        self.closed = self.sealed = self.transferred = self.transfer_started = False
        self.error: Any = None
        stack.push(self.close)
        self.stream = (output / HISTORY_NAME).open('x', encoding='utf-8')
        self.witness = witness.install(self.scope, journal)
        if probability_factory is not None:
            self.probability = probability_factory(self.scope)

    def observe(self, pipe: Any, frame: int) -> None:
        try:
            if self.closed or self.sealed or self.error is not None:
                raise ValueError('early_history_closed')
            if self.count >= len(self.frames) or frame != self.frames[self.count]:
                raise ValueError('early_history_frame_gap')
            if self.pipe is not None and self.pipe is not pipe:
                raise ValueError('early_history_pipe_changed')
            self.pipe = pipe
            pair = self.reader.read_pair(self.witness, self.journal, pipe, frame)
            rows = json.loads(pair.rows_json)
            if len(rows) != 2: raise ValueError('early_history_pair_missing')
            payload = dict(frame=frame, holds=pair.holds, rows=[compact(row) for row in rows])
            proof = None if self.probability is None else self.probability.snapshot(rows[1])
            if proof is not None: payload['probability'] = proof
            value = encoded(payload)
            self.stream.write(value + '\n')
            self.stream.flush()
            self.records.append(value)
            self.digest.update((value + '\n').encode())
            if proof is not None and self.probability.is_candidate(proof):
                saved = encoded(proof)
                self.probability_records.append(saved)
                self.probability_digest.update((saved + '\n').encode())
            self.count += 1
            self.last_frame, self.last_steps = frame, self.journal.steps
        except BaseException as error:
            self.error = error
            raise

    def seal(self, frame: int) -> None:
        if self.closed or self.sealed or self.error is not None or self.last_frame != frame:
            raise ValueError('early_history_seal_boundary')
        if self.journal.active is not None or self.journal.steps != self.last_steps:
            raise ValueError('early_history_seal_J_changed')
        self.scope.close()  # 早期PBとStreamを解放してから通常Sessionの取得器生成を許す。
        self.stream.close()
        self.probability_records = tuple(self.probability_records)
        self.sealed = True

    def probability_inputs(self) -> tuple[str, ...]:
        """移管済みの保存候補だけ返す。基準登録や現在値の資格は付与しない。"""
        if self.closed or not self.sealed or not self.transferred or self.error is not None:
            raise ValueError('early_probability_inputs_lifetime')
        payload = ''.join(value + '\n' for value in self.probability_records).encode()
        if hashlib.sha256(payload).digest() != self.probability_digest.digest():
            raise ValueError('early_probability_inputs_changed')
        return tuple(self.probability_records)

    def prime(self, capture: Any, frame: int) -> dict:
        if self.closed or not self.sealed or self.transfer_started or self.error is not None:
            raise ValueError('early_history_transfer_lifetime')
        if (capture.journal is not self.journal or capture.pipe is not self.pipe
                or capture.witness.journal is not self.journal or capture.witness.closed
                or capture.witness.error is not None or capture.error is not None
                or capture.closed or capture.last_frame != -1 or capture.scopes):
            raise ValueError('early_history_transfer_owner')
        if (frame != self.last_frame or self.journal.history.frame != frame
                or self.journal.closed or self.journal.errors or self.journal.steps != self.last_steps):
            raise ValueError('early_history_transfer_clock')
        payload = ''.join(value + '\n' for value in self.records).encode()
        if hashlib.sha256(payload).digest() != self.digest.digest():
            raise ValueError('early_history_buffer_changed')
        if hashlib.sha256((self.output / HISTORY_NAME).read_bytes()).digest() != self.digest.digest():
            raise ValueError('early_history_saved_changed')
        self.transfer_started = True  # 途中失敗後も同一履歴を再消費しない。
        try:
            for raw in self.records:
                for item in json.loads(raw)['rows']:
                    capture.consume(item['row'], available_frame=frame)
            receipt = dict(kind='origin_history_primed', recorded_first=self.frames[0],
                recorded_last=frame, available_frame=frame, updates=self.count,
                source_sha256=self.digest.hexdigest(), physical_identity_verified=False,
                quality_gate_clear=False, future_fire_power_supply_authorized=False)
            capture.stream.write(encoded(receipt) + '\n')
            capture.stream.flush()
            capture.last_frame = frame
            self.transferred = True
            self.records.clear()
            return receipt
        except BaseException as error:
            self.error = capture.error = error
            capture.record_failure(frame, None, error)
            raise

    def close(self, kind: Any, body: Any, trace: Any) -> bool:
        errors = []
        try:
            self.scope.__exit__(kind, body, trace)
        except BaseException as error:
            errors.append(repr(error))
        if self.stream is not None:
            try: self.stream.close()
            except BaseException as error: errors.append(repr(error))
        receipt = dict(updates=self.count, last_frame=self.last_frame, sealed=self.sealed,
            probability_enabled=self.probability is not None,
            probability_candidates=len(self.probability_records), probability_sha256=self.probability_digest.hexdigest(),
            transfer_started=self.transfer_started, transferred=self.transferred, source_sha256=self.digest.hexdigest(),
            original_error=None if body is None else repr(body), cleanup_errors=errors,
            error=None if self.error is None else repr(self.error), quality_gate_clear=False)
        try:
            with (self.output / CLOSE_NAME).open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream)
        except BaseException as error: errors.append(repr(error))
        self.closed = True
        self.records.clear()
        self.probability_records = ()
        self.probability = None
        self.journal = self.pipe = self.witness = self.reader = self.stream = None
        if body is None and errors: raise ValueError('early_history_cleanup:' + repr(errors))
        if body is None and self.error is not None: raise self.error
        return False
