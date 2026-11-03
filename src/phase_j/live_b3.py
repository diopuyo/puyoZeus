"""既定OFFのB3接続。認識観測は通知へ、勝率は公開値だけへ接続する。"""
from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter, thread_time
from typing import Any

from src.b3_contents import contents_from_row
from src.b3_live_display import B3Display
from src.b3_realtime import RealtimeRows
from src.b3_timing import TimingTable
from src.b3_train_model import load_model
from src.production_config import DISPLAY_FAST_EMA_DEFAULT

ENV_ENABLED = 'PUYO_B3_JUDGMENT'
DEFAULT_MODEL = Path('models/b3/model.joblib')
DEFAULT_TABLE = Path('models/b3/timing_table.json')
MS = 1000.0


def display_assets(options: Any, assets: dict[str, str]) -> dict[str, str]:
    """公開DTOのモデル識別子を、実際に表示へ使うモデルへ合わせる。"""
    if not getattr(options, 'b3_judgment', False):
        return assets
    from src.b3_source import sha256_file
    model = options.b3_model
    if getattr(options, 'live_24h', False) and getattr(options, 'live_24h_ama', True):
        model = options.live_24h_model
    return dict(assets, prediction_model_hash='sha256:'+sha256_file(model))


def configure(options: Any, parser: Any) -> None:
    """明示指定時だけ起動。spawn認識workerにも同じON/OFFを配送する。"""
    from .live_24h_config import configure as configure_24h
    configure_24h(options, parser)
    from .live_fire_board_config import configure as configure_fire_board
    configure_fire_board(options, parser)
    from .live_obs_trial import configure as configure_trial
    configure_trial(options, parser)
    trial = getattr(options, 'live_v49f', False)
    if type(trial) is not bool or (trial and not (options.b3_judgment and options.b3_async)):
        parser.error('live_v49fはboolで、trueにはb3_judgmentとb3_asyncが必要です')
    os.environ['PUYO_LIVE_V49F'] = str(int(trial))
    if type(options.b3_judgment) is not bool:
        parser.error('b3_judgmentはboolです')
    if type(getattr(options, 'b3_async', False)) is not bool:
        parser.error('b3_asyncはboolです')
    if getattr(options, 'b3_async', False) and not options.b3_judgment:
        parser.error('b3_asyncにはb3_judgmentが必要です')
    for name in ('display_completion_driven', 'display_fast_ema'):
        value = getattr(options, name, False)
        if type(value) is not bool:
            parser.error(f'{name}はboolが必要です')
        setattr(options, name, value and options.b3_judgment)
    os.environ[ENV_ENABLED] = '1' if options.b3_judgment else '0'
    if not options.b3_judgment:
        return
    if not options.split_evaluation or options.worker_mode != 'process' or options.compare:
        parser.error('B3はprocess・split-evaluationのライブ経路で使用してください')
    for name in ('b3_model', 'b3_timing_table'):
        value = Path(getattr(options, name)).resolve()
        if not value.is_file():
            parser.error(f'{name}がありません: {value}')
        setattr(options, name, value)


def prefetch_warning(pipe: Any, frame: Any) -> bool:
    """b3_warning_prefetch (試作・既定OFF): 盤面認識の前に、同じコマの予告読取を worker で始める。

    今の入力世代の観測器がまだ無い (最初のコマ・世代切替) ときは何もしない。そのコマは
    capture() の中で従来どおり同期で読む。結果は同じ関数・同じ画素なので値は変わらない。
    """
    if os.environ.get(ENV_ENABLED) != '1':
        return False
    owner = getattr(pipe, 'pipe', pipe)
    epoch = (getattr(pipe, 'epoch', None), getattr(pipe, 'publishing_ready', True))
    observation = getattr(owner, '_b3_capture', None)
    if observation is None or getattr(owner, '_b3_epoch', None) != epoch:
        return False
    return observation.prefetch_warning(frame.image)


def capture(pipe: Any, result: Any, frame: Any, counts: tuple[int, int]) -> tuple | None:
    """認識processで画像の生存中に取得する。OFFなら追加読取・import無し。"""
    if os.environ.get(ENV_ENABLED) != '1':
        return None
    from src.row1_publication import evaluation_result
    result = evaluation_result(result)
    from src.b3_observation import ObservationCapture
    from src.b3_live_queues import LiveQueues
    owner = getattr(pipe, 'pipe', pipe)
    epoch = (getattr(pipe, 'epoch', None), getattr(pipe, 'publishing_ready', True))
    if not hasattr(owner, '_b3_capture') or owner._b3_epoch != epoch:
        prepared = getattr(owner, '_b3_prepared_capture', None)
        owner._b3_prepared_capture = None
        close_capture(owner)
        from .live_speed import observation_options
        owner._b3_capture = prepared if prepared is not None else ObservationCapture(**observation_options())
        owner._b3_queues = LiveQueues()
        owner._b3_epoch = epoch
    from .live_score_share import shared_scores
    # 入力較正の確定枠だけを共有する。B3の評価・重み・採用条件は変えない。
    alignment = getattr(owner, '_score_roi_alignment', None)
    if alignment is not None:
        owner._b3_capture.ocr._region_offsets = dict(owner._score_ocr._region_offsets)
    scores = shared_scores(owner, owner._b3_capture.ocr, frame)
    options = {} if scores is None else dict(scores=scores)
    row, events = owner._b3_capture.update(result, frame.image, frame.index, frame.media_sec, counts, **options)
    from .live_stage_timing import measure
    queues = measure('b3_queues', owner._b3_queues.update,
                     owner, result, frame.image, frame.index, frame.media_sec, counts)
    row['adopted_game'] = owner._b3_queues.current_game
    if os.environ.get('PUYO_LIVE_V49F') == '1':
        from .live_v49f_capture import CaptureV49F
        if not hasattr(owner, '_v49f_capture') or owner._v49f_epoch != epoch:
            prepared = getattr(owner, '_v49f_prepared_capture', None)
            owner._v49f_prepared_capture = None
            factory = CaptureV49F
            if os.environ.get('PUYO_OBS_PREPARE_EXACT') == '1':
                from .live_obs_prepare import create_v49f_capture
                factory = create_v49f_capture
            owner._v49f_capture, owner._v49f_epoch = prepared if prepared is not None else factory(), epoch
        scores = scores if scores is not None else owner._b3_capture.ocr.read(frame.image)
        row['_v49f'] = measure('v49f_capture', owner._v49f_capture.capture, result, frame, row, scores)
        row['_v49f']['input_epoch'] = epoch
    if os.environ.get('PUYO_LIVE_AMA_PLAIN') == '1':
        from .live_v49f_capture import raw_side
        row['_ama'] = dict(t=frame.media_sec, game=row['game_idx'],
                           raw=[raw_side(s) for s in (result.p1, result.p2)])
    _capture_24h(owner, result, frame, row, epoch)
    return row, events, queues, epoch


def _capture_24h(owner: Any, result: Any, frame: Any, row: dict, epoch: Any) -> None:
    """OFFは追加の観測・importをしない。"""
    if os.environ.get('PUYO_LIVE_24H') == '1' and os.environ.get('PUYO_LIVE_24H_AMA', '1') == '1':
        from .live_ex_capture import capture as capture_ex
        capture_ex(owner, result, frame, row, epoch)


def close_capture(pipe: Any) -> None:
    """機器世代の終了・入力終了で予告CNN workerを回収する。"""
    owner = getattr(pipe, 'pipe', pipe)
    for capture in (getattr(owner, '_b3_capture', None), getattr(owner, '_b3_prepared_capture', None)):
        close = getattr(capture, 'close', None)
        if close is not None:
            close()
    if hasattr(owner, '_b3_prepared_capture'):
        owner._b3_prepared_capture = None


def prepare_capture(pipe: Any) -> None:
    """試験版だけ、画素を取得する前に不変モデルをロードする。観測は流さない。"""
    if os.environ.get('PUYO_LIVE_SCORE_CADENCE') != '1' or os.environ.get(ENV_ENABLED) != '1':
        return
    started = perf_counter()
    # LiveSnapshotInputs.boundaryの初回importも認識フレーム外で済ませる。
    from scripts import visualize_advantage_overlay
    from src.b3_observation import ObservationCapture
    from .live_speed import observation_options
    owner = getattr(pipe, 'pipe', pipe)
    if getattr(owner, '_b3_prepared_capture', None) is None:
        owner._b3_prepared_capture = ObservationCapture(**observation_options())
        print('B3_PREPARE '+json.dumps(dict(phase='before_capture',
            wall_ms=(perf_counter()-started)*MS)), flush=True)


def reference_state(tracker: Any) -> dict:
    """死亡の本番値と物理合図だけを毎通知で取得する。"""
    from src.b3_stage2_trace import physical_times
    records = list(getattr(tracker, 'records', ()))
    if tracker.current is not None:
        records.append(tracker.current)
    from src.production_config import LIVE_REFERENCE_TIMES_ONLY_DEFAULT
    enabled = os.environ.get('PUYO_LIVE_REFERENCE_TIMES_ONLY',
                             str(int(LIVE_REFERENCE_TIMES_ONLY_DEFAULT))) == '1'
    mappings = ([reference_times(r) for r in records] if enabled else
                [r if isinstance(r, dict) else asdict(r) for r in records])
    return dict(source=tracker.source, probability=tracker.probability,
                physical=physical_times(mappings).tolist())


def reference_times(record: Any) -> dict:
    """時刻だけを投影する。値・監査・盤面の深い複製や履歴キャッシュは不要。"""
    def field(value: Any, key: str, default: Any = None) -> Any:
        return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)

    def pair(value: Any, names: tuple[str, str]) -> dict:
        return {name: field(value, name) for name in names}

    return dict(pair(record, ('trigger_sec', 'closed_sec')),
        chains=[pair(chain, ('observed_sec', 'end_signal_sec'))
                for chain in field(record, 'chains', ())],
        landings=[pair(landing, ('fall_start_sec', 't_sec'))
                  for landing in field(record, 'landings', ())])


class LiveB3:
    """状態は全通知で更新し、Rの死亡証明だけを表示契約へ渡す。"""

    def __init__(self, model: Path, table: Path, output: Path,
                 async_enabled: bool = False, completion_driven: bool = False,
                 fast_ema: bool = DISPLAY_FAST_EMA_DEFAULT, trial_helpers: bool = True) -> None:
        from src.b3_column_spec import COLUMN_NAMES
        if os.environ.get('PUYO_B3_M3') == '1':
            from src.b3_column_spec import build_column_specs
            COLUMN_NAMES = tuple(c.name for c in build_column_specs(True))
        from src import puyo_core_bridge as native
        if not native.NATIVE_AVAILABLE or not hasattr(native._native, 'b3_search_py'):
            raise RuntimeError('B3探索にはc4ebcfb版のpuyo_coreが必要です')
        from src.b3_ensemble import ensemble_enabled, load_ensemble_model
        # 既定OFFは従来どおり load_model (1 本)。b3_ensemble ON のときだけ、マニフェストから複数モデルの平均の入れ物を作る。
        self.model = load_ensemble_model(model) if ensemble_enabled() else load_model(model)
        for evaluator in (self.model.static, self.model.exchange):
            if evaluator is None or set(evaluator.columns) - set(COLUMN_NAMES):
                raise ValueError('B3モデルの評価器または特徴列が実装と一致しません')
        if os.environ.get('PUYO_B3_M3') == '1':
            from .live_obs_trial import c2_enabled, check_c2_model
            if c2_enabled():   # 予告アイコンの列が残っている / 正味の列が無いモデルは C2 として使わない
                check_c2_model(self.model)
        self.table = TimingTable.from_json(table.read_text(encoding='utf-8'))
        self.rows = RealtimeRows(self.table)
        self.fast_ema = fast_ema
        self.display = self._new_display()
        self.epoch: Any = None
        self.physical: set[float] = set()
        self.physical_count = 0
        self.generation, self.result_frame = 0, -1
        self.async_failed = False
        self.completion_driven = completion_driven
        self.adopted_game: int | None = None
        self.async_engine = None
        self._setup_trial(model, table, output, trial_helpers)
        self._open_pipeline(model, table, output, async_enabled)

    def _open_pipeline(self, model: Path, table: Path, output: Path, async_enabled: bool) -> None:
        """24hのEX起動後にB3や記録の起動が失敗しても、自分の子を残さない。"""
        try:
            if async_enabled:
                from src.b3_async import AsyncB3
                self.async_engine = AsyncB3(model, self.table, output=output,
                                           include_rows=self.time_gate is not None)
            from .live_obs_audit import open_audit
            self.audit = open_audit(output / 'b3_updates.jsonl')
            self.state_audit = None if async_enabled else open_audit(output / 'b3_state_timing.jsonl')
            self._write_config(model, table, output, async_enabled)
        except BaseException:
            if getattr(self, 'live24h', None) is not None:
                self._close_failed_setup()
            raise

    def _close_failed_setup(self) -> None:
        from contextlib import suppress
        for name in ('live24h', 'async_engine', 'audit', 'state_audit'):
            resource = getattr(self, name, None)
            if resource is not None:
                with suppress(Exception):
                    resource.close()

    def _setup_trial(self, model: Path, table: Path, output: Path, trial_helpers: bool) -> None:
        """試験欄だけの所有者。OFF時は資産読取・process起動を行わない。"""
        self.v49f = None
        self.ama = self.v49f_baseline = None
        self.time_gate = None
        self.live24h = None
        if trial_helpers and os.environ.get('PUYO_LIVE_24H') == '1':
            from .live_24h import Live24h
            self.live24h = Live24h(output)
        if os.environ.get('PUYO_B3_TIME_GATE') == '1':
            from .live_b3_time_gate import TimeGate
            self.time_gate = TimeGate(output)
        if trial_helpers and self.live24h is None and os.environ.get('PUYO_LIVE_AMA_PLAIN') == '1':
            from .live_ama import LiveAma
            self.ama = LiveAma(output)
        if trial_helpers and self.live24h is None and os.environ.get('PUYO_LIVE_V49F') == '1':
            from .live_v49f import LiveV49F
            if os.environ.get('PUYO_B3_M3') == '1':
                from .live_obs_trial import create_baseline
                self.v49f_baseline = create_baseline(table, output, self.fast_ema)
                self.v49f = LiveV49F(Path('models/v49f/b3_v6.joblib'), table, output)
            else:
                self.v49f = LiveV49F(model, table, output)

    def _write_config(self, model: Path, table: Path, output: Path, async_enabled: bool) -> None:
        from src.b3_source import sha256_file
        from src.b3_search import search_manifest
        value = dict(model=str(model),
            model_sha256=sha256_file(model), timing_table=str(table),
            timing_sha256=sha256_file(table), row_builder='src.b3_rows._make_row',
            search=search_manifest(), async_enabled=async_enabled, completion_driven=self.completion_driven,
            fast_ema=self.fast_ema, reference_death='production_R_F10')
        if os.environ.get('PUYO_B3_M3') == '1':
            from src import b3_trial_config
            from .live_obs_trial import c2_enabled, c2_record
            flags = {k:getattr(b3_trial_config,k) for k in sorted(b3_trial_config.M3_FLAGS)}
            if c2_enabled():   # C2 のときだけ別の記録。M3 の記録 (キーと順序) は従来のまま
                value.update(c2_record(self.model), ledger_flags=flags)
                self._record_ensemble(value)
            else:
                value.update(trial_model='M3',feature_columns=295,ledger_flags=flags)
        if getattr(self, 'live24h', None) is not None:
            self.live24h.record_config(value)
        (output / 'b3_config.json').write_text(json.dumps(value,indent=2),encoding='utf-8')

    def _record_ensemble(self, value: dict) -> None:
        """b3_ensemble ON のときだけ、平均の仕方と各モデルの SHA256 を b3_config.json の末尾へ足す。OFF は何も足さない。"""
        from src.b3_ensemble import ensemble_enabled, ensemble_record
        if ensemble_enabled():
            value.update(ensemble_record(self.model))

    def _new_display(self) -> B3Display:
        """試合・採用開始・入力世代のリセットでも明示設定を維持する。"""
        return B3Display(fast_ema=getattr(self, 'fast_ema', DISPLAY_FAST_EMA_DEFAULT))

    def _prepare_frame(self, frame: dict, epoch: Any) -> None:
        """従来と同じ順に世代・表示の境界を進める。"""
        if epoch != self.epoch:
            self.rows, self.display = RealtimeRows(self.table), self._new_display()
            self.epoch = epoch
            self.generation += 1
            self.physical.clear()
            self.physical_count = 0
            self.result_frame = -1
            self.adopted_game = None
        adopted = frame.get('adopted_game')
        if adopted is not None and adopted != self.adopted_game:
            self.adopted_game = adopted
            self.generation += 1
            self.display = self._new_display()
        previous_game = self.display.game
        self.display.boundary(frame['game_idx'])
        self.display.chain.observe(frame)
        if previous_game != self.display.game:
            self.physical.clear()
            self.physical_count = 0
            self.result_frame = -1

    def submit_ex_early(self, notice: Any) -> None:
        """評価のイベント待ちより先に、同じ通知・世代で一度だけ要求する。"""
        unified = getattr(self, 'live24h', None)
        if unified is None or not unified.settings.get('early_submit', False) or not unified.ama:
            return
        if notice.b3 is None:
            raise ValueError('B3有効時の認識観測がありません')
        if getattr(self, '_early_ex_notice', None) is not None:
            raise RuntimeError('EX先行要求の通知が未処理です')
        frame, _, _, epoch = notice.b3
        changed = epoch != self.epoch
        adopted = None if changed else self.adopted_game
        generation = self.generation + int(changed)
        generation += int(frame.get('adopted_game') is not None and frame['adopted_game'] != adopted)
        from .live_stage_timing import measure
        measure('ex_early_submit', unified.update, frame, generation, notice.captured_at)
        self._early_ex_notice = notice
        self._early_ex_generation = generation

    def _update_ex(self, notice: Any, frame: dict) -> None:
        """先行要求済みなら二重送信せず、世代の一致を確かめる。"""
        early = getattr(self, '_early_ex_notice', None)
        if early is not None:
            if early is not notice or self._early_ex_generation != self.generation:
                raise RuntimeError('EX先行要求とB3通知の世代が不一致です')
            self._early_ex_notice = None
        elif getattr(self, 'live24h', None) is not None:
            self.live24h.update(frame, self.generation, notice.captured_at)

    def update(self, notice: Any, overlay: Any) -> None:
        if notice.b3 is None:
            raise ValueError('B3有効時の認識観測がありません')
        frame, events, queues, epoch = notice.b3
        self.latest_frame = frame
        self.latest_overlay = overlay
        self._prepare_frame(frame, epoch)
        self._update_ex(notice, frame)
        if self.async_engine is not None:
            self._async_update(frame, events, queues, notice)
        else:
            row = self.rows.update(frame, events, queues)
            self.state_audit.write(json.dumps(dict(frame=frame['frame'], t_sec=frame['t_sec'],
                epoch=self.generation, game=frame['game_idx'], state_wait_ms=0.,
                trigger=self.rows.trigger if row is not None else 0, **self.rows.state_timings))+'\n')
            if row is not None:
                self._predict(row, notice)
        self._display(frame, overlay)
        if getattr(self, 'v49f', None) is not None:
            from .live_stage_timing import measure
            measure('v49f_submit', self.v49f.update, frame, self.generation)
        if getattr(self, 'ama', None) is not None:
            from .live_stage_timing import measure
            measure('ama_submit', self.ama.update, frame, self.generation)
        if getattr(self, 'v49f_baseline', None) is not None:
            from .live_obs_trial import baseline_notice
            self.v49f_baseline.update(baseline_notice(notice), overlay)

    def _async_update(self, frame: dict, events: tuple, queues: tuple, notice: Any) -> None:
        if self.async_failed:
            return
        if not getattr(self, 'completion_driven', False):
            self._poll_results(frame)
        if not self.async_failed:
            b3_frame = {k: v for k, v in frame.items() if k not in ('_v49f', '_ama', '_ex')}
            unified = getattr(self, 'live24h', None)
            options = dict(features_only=True) if unified is not None and unified.ama else {}
            self.async_engine.submit(b3_frame, events, queues, self.generation,
                                     captured_at=notice.captured_at, **options)

    def _poll_results(self, frame: dict) -> bool:
        """受理済み結果または失敗状態の更新だけを完了として返す。"""
        before = (self.result_frame, self.async_failed)
        for result in self.async_engine.poll():
            if 'fatal' in result:
                self.async_failed = self.display.failed = True
                self.audit.write(json.dumps(result)+'\n')
            elif 'done' not in result:
                self._accept_result(result, frame)
        return before != (self.result_frame, self.async_failed)

    def receive_completed(self) -> bool:
        """完了通知で起床した評価所有者から呼ぶ。新たな認識や推論は行わない。"""
        if not hasattr(self, 'latest_frame'):
            return False
        changed = self._poll_results(self.latest_frame)
        if changed:
            self._display(self.latest_frame, self.latest_overlay)
        return changed

    def _accept_result(self, result: dict, frame: dict) -> None:
        from src.b3_eval_adapter import MAX_HOLD_SEC
        if 'row' in result:
            result['mode'] = None if result['row'] is None else result['row']['mode_exchange']
        valid = (result['epoch'] == self.generation and result['game'] == frame['game_idx']
                 and self.result_frame < result['frame'] <= frame['frame']
                 and 0 <= frame['t_sec']-result['t_sec'] <= MAX_HOLD_SEC)
        if valid and result['error'] is None and getattr(self, 'live24h', None) is not None:
            result = self._predict_24h(result)
        if getattr(self, 'time_gate', None) is not None:
            self.time_gate.result(result, valid)
        if valid:
            self.result_frame = result['frame']
            if result['error'] is None:
                self.display.prediction(result['raw_probability'], result['mode'], result['t_sec'])
                self.display.contents = result.get('contents')
            else:
                self.display.failed = True
        costs = dict(result['timings'])
        for key, value in result['state_timings'].items():
            costs[key] = costs.get(key, 0.) + value
        costs.setdefault('inference_ms', 0.)
        result_wait = (perf_counter()-result['completed_at'])*MS + result.get('feature_result_wait_ms', 0.)
        from .live_stage_timing import b3_result
        b3_result(result['frame'], result['t_sec'], result['captured_at'], costs,
                  epoch=result['epoch'], accepted=valid, result_wait_ms=result_wait,
                  state_wait_ms=result['state_wait_ms'], compute_wait_ms=result['compute_wait_ms'])
        self.audit.write(json.dumps(dict(frame=result['frame'], t_sec=result['t_sec'],
            epoch=result['epoch'], game=result['game'], trigger=result['trigger'],
            feature_ms=sum(costs.get(k, 0.) for k in ('state_ms', 'ledger_ms', 'search_ms', 'feature_only_ms')),
            **costs, state_wait_ms=result['state_wait_ms'], compute_wait_ms=result['compute_wait_ms'],
            result_wait_ms=result_wait, wait_ms=result['state_wait_ms']+result['compute_wait_ms']+result_wait,
            raw_probability=result['raw_probability'], captured_at=result['captured_at'],
            completed_at=result['completed_at'], error=result['error'], accepted=valid))+'\n')

    def _predict_24h(self, result: dict) -> dict:
        """C2の生値を計算せず、選択したモデルの失敗も既存の記録口へ返す。"""
        if not self.live24h.ama:
            return result
        started, cpu_started = perf_counter(), thread_time()
        value = dict(result)
        try:
            value['raw_probability'] = self.live24h.predict(result)
        except Exception as exc:
            value['error'] = f'{type(exc).__name__}: {exc}'
        completed = perf_counter()
        value.update(completed_at=completed,
                     feature_result_wait_ms=(started-result.get('completed_at', started))*MS,
                     timings=dict(result.get('timings', {}), inference_ms=(completed-started)*MS,
                                  inference_cpu_ms=(thread_time()-cpu_started)*MS))
        return value

    def _predict(self, row: dict, notice: Any) -> None:
        import math
        started = perf_counter()
        cpu_started = thread_time()
        error = None
        try:
            p = self.model.predict_row(row)
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('B3の予測が有限な確率ではありません')
            self.display.prediction(p, int(row['mode_exchange']), row['t_sec'])
            self.display.contents = contents_from_row(row)
            self.result_frame = int(row['frame_idx'])
        except Exception as exc:
            self.display.failed = True
            error = f'{type(exc).__name__}: {exc}'
            logging.getLogger(__name__).exception('B3推論失敗')
        from .live_stage_timing import b3_result
        b3_result(notice.frame, notice.t_sec, notice.captured_at,
                  dict(self.rows.timings, inference_ms=(perf_counter()-started)*MS),
                  epoch=self.generation, accepted=error is None)
        self.audit.write(json.dumps(dict(frame=notice.frame, t_sec=notice.t_sec,
            trigger=row['trigger_mask'], feature_ms=self.rows.feature_ms,
            **self.rows.timings, state_wait_ms=0., compute_wait_ms=0., result_wait_ms=0., wait_ms=0.,
            inference_ms=(perf_counter()-started)*MS, raw_probability=self.display.probability,
            inference_cpu_ms=(thread_time()-cpu_started)*MS,
            captured_at=notice.captured_at, completed_at=perf_counter(), error=error))+'\n')

    def _display(self, frame: dict, overlay: Any) -> None:
        from scripts.visualize_advantage_overlay import _WINPROB_K
        if getattr(self, 'live24h', None) is not None:
            self.live24h.revalidate(self.display)
        if getattr(self, 'completion_driven', False):
            self._restore_display_tick(frame['t_sec'])
        reference = getattr(overlay, 'b3_reference', None)
        if reference is None:
            reference = reference_state(overlay.tracker)
        for stamp in reference['physical']:
            if stamp <= frame['t_sec'] and stamp not in self.physical:
                self.physical.add(float(stamp))
                self.physical_count += 1
        self.display.update(frame['t_sec'], reference['source'], reference['probability'],
            self.physical_count, _WINPROB_K, bool(frame['match_active']))

    def _restore_display_tick(self, t_sec: float) -> None:
        """同じ認識時刻の完了はその時刻のEMAを置換し、余分な平滑ステップを作らない。"""
        tick = (self.display, t_sec)
        if getattr(self, '_display_tick', None) != tick:
            self._display_tick = tick
            self._smoother_before_tick = deepcopy(self.display.smoother)
        else:
            self.display.smoother = deepcopy(self._smoother_before_tick)

    def decorate(self, row: dict) -> None:
        adv, probability = self.display.shown
        if getattr(self, 'completion_driven', False):
            row['judgment_id'] = (self.generation, self.result_frame, self.display.source,
                                  self.display.raw, row.get('match_active', True), self.display.game)
        row.update(probability=probability, advantage=adv, raw_probability=self.display.raw,
            calibration_id='b3-stage2-display',
            source=self.display.source, hold=self.display.raw is None,
            display_layers=dict(current_p1=self.display.raw,
                predicted_p1=None, displayed_p1=probability if self.display.raw is not None else None,
                includes_prediction=False, prediction_discard_reason=None), prediction_sources=[])
        if self.display.source in ('static', 'exchange'):
            if self.display.contents is not None:
                row['b3_contents'] = self.display.chain.decorate(self.display.contents)
            row.update(b3_state_t_sec=self.display.last_success,
                       b3_state_frame=self.display.contents['frame'] if self.display.contents
                       else getattr(self, 'result_frame', None))
        if getattr(self, 'v49f', None) is not None:
            baseline = getattr(self, 'v49f_baseline', None)
            self.v49f.decorate(row, baseline.display.raw if baseline is not None else None,
                                separate_baseline=baseline is not None)
            if getattr(self.v49f, 'direct', False) and 'judgment_id' in row:
                # 段1 (既定OFF): X の初出だけで判定IDを変え、既存の重複排除のまま即公開させる。
                row['judgment_id'] = (*row['judgment_id'], self.v49f.x_completed_at)
        if getattr(self, 'ama', None) is not None:
            self.ama.decorate(row)
        if os.environ.get('PUYO_B3_M3') == '1':
            from .live_obs_trial import trial_label
            row['b3_model_label'] = trial_label()
        if getattr(self, 'time_gate', None) is not None:
            self.time_gate.publication(row, self.async_engine)
        if getattr(self, 'live24h', None) is not None:
            self.live24h.decorate(row)

    def close(self) -> None:
        """入力終了後に子processと監査ファイルを回収する。"""
        if getattr(self, 'v49f', None) is not None:
            self.v49f.close()
        for name in ('ama', 'v49f_baseline'):
            if getattr(self, name, None) is not None:
                getattr(self, name).close()
        if self.async_engine is not None:
            self.async_engine.close()
            for result in self.async_engine.tail:
                if 'fatal' in result:
                    self.audit.write(json.dumps(result)+'\n')
                elif hasattr(self, 'latest_frame'):
                    self._accept_result(result, self.latest_frame)
        if self.state_audit is not None:
            self.state_audit.close()
        if getattr(self, 'live24h', None) is not None:
            self.live24h.close()
        self.audit.close()
        if getattr(self, 'time_gate', None) is not None:
            self.time_gate.close()
