"""同一区間の実ライブを計装する。評価値・認識入力は変更しない。"""
from __future__ import annotations

from collections import Counter
from functools import wraps
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

# 8fe251cの展開先でも同じ計装器を使い、対象コードはcwdから読む。
sys.path.insert(0, str(Path.cwd()))
SAMPLE_NOTICES = 30
MILLISECONDS = 1000.0
MODEL_CALLS: Counter[str] = Counter()
STREAMS: dict[str, Any] = {}
TIMINGS: Counter[str] = Counter()
STACK: list[list[float]] = []


def timed(original: Any, label: Any) -> Any:
    """親子呼出の重複を除いた壁時計時間を、通知ごとに集計する。"""
    @wraps(original)
    def call(*args: Any, **kwargs: Any) -> Any:
        name = label(*args, **kwargs) if callable(label) else label
        frame = [perf_counter(), 0.0]
        STACK.append(frame)
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = perf_counter()-frame[0]
            STACK.pop()
            TIMINGS[name] += (elapsed-frame[1])*MILLISECONDS
            if STACK:
                STACK[-1][1] += elapsed
    return call


def instrument_breakdown() -> None:
    from src.exchange_event_overlay import ExchangeEventOverlay as Overlay
    from src.exchange_event_landing import ExchangeLandingProjection
    from src.exchange_event_m0 import FileM0Predictor
    from src.exchange_post_counter_bound import PostCounterDeathBound
    for owner, names in ((Overlay, ('_refresh_features', '_static', '_deliver_fires',
            '_observe_signals', '_observe_predictions', '_observe_prefire', '_observe_guards')),
            (ExchangeLandingProjection, ('update', '_landing_gfe', '_evaluate')),
            (PostCounterDeathBound, ('prove',)), (FileM0Predictor, ('__call__',))):
        for name in names:
            if hasattr(owner, name):
                setattr(owner, name, timed(getattr(owner, name), owner.__name__+'.'+name))


def configure_comparison() -> None:
    """LR対照だけ静止G_feの差替引数を外し、その他の本番構成を保つ。"""
    if os.environ.get('RT_STATIC_LR') != '1':
        return
    import shlex
    from src import production_config
    original = production_config.exchange_event_flags

    def flags() -> str:
        args = shlex.split(original())
        flag = '--exchange-event-static-gfe-dir'
        if flag in args:
            index = args.index(flag)
            del args[index:index+2]
        return shlex.join(args)

    production_config.exchange_event_flags = flags


def instrument_static_builder() -> None:
    """追加診断時だけ、workerの特徴要求と親の特徴探索を対応付ける。"""
    if os.environ.get('RT_DETAIL_PROFILE') != '1':
        return
    from scripts import visualize_advantage_overlay as render
    original = render._exchange_static_input

    @wraps(original)
    def build(*args: Any, **kwargs: Any) -> Any:
        import cProfile
        import pstats
        profile = cProfile.Profile()
        started = perf_counter()
        value = profile.runcall(original, *args, **kwargs)
        elapsed = perf_counter()-started
        stats = pstats.Stats(profile)
        functions = sorted(stats.stats.items(), key=lambda row: row[1][3], reverse=True)
        emit('static_profile', dict(elapsed_sec=args[2], milliseconds=elapsed*MILLISECONDS,
            functions=[dict(file=key[0], line=key[1], name=key[2], calls=data[1],
                            self_ms=data[2]*MILLISECONDS, cumulative_ms=data[3]*MILLISECONDS)
                       for key, data in functions[:20]]))
        return value

    render._exchange_static_input = build


def emit(name: str, row: dict) -> None:
    directory = Path(os.environ['RT_VERIFY_OUTPUT'])
    directory.mkdir(parents=True, exist_ok=True)
    key = f'{name}-{os.getpid()}'
    if key not in STREAMS:
        STREAMS[key] = (directory/(key+'.jsonl')).open('a', encoding='utf-8', buffering=1)
    STREAMS[key].write(json.dumps(row, ensure_ascii=False)+'\n')


def configuration(overlay: Any) -> dict:
    projection, tracker = overlay._landing_projection, overlay.tracker
    bound = getattr(projection, 'post_counter_bound', None)
    history = getattr(overlay, '_history_integrity', None)
    models = tracker.models
    landing = getattr(models, 'landing_models', None) or models
    return dict(death_input_integrity=getattr(projection, 'death_input_integrity', False),
        score_attribution_v2=getattr(tracker, 'score_attribution_v2', False),
        post_counter_current_puyo_integrity_v2c=getattr(bound, 'current_puyo_integrity_v2c', False),
        death_history_integrity=history is not None,
        death_history_boundary_start=getattr(history, 'boundary_score_start', False),
        post_counter_completion_recent_integrity=getattr(bound, 'completion_recent_integrity', False),
        death_integrity_v2h2=bool(getattr(history, 'strict_integrity', False)
                                  and getattr(bound, 'strict_integrity', False)),
        death_freshness_cap=getattr(bound, 'cap_completed', False),
        static_gfe=type(models.estimators['G_fe']).__name__,
        landing_gfe=type(landing.estimators['G_fe']).__name__)


def instrument_models() -> None:
    from src.exchange_event_evaluator import FileExchangeModels
    original = FileExchangeModels.predict_source_probability

    @wraps(original)
    def predict(self: Any, model_name: str, features: Any) -> float:
        value = timed(original, 'model:'+model_name+':'+type(self.estimators[model_name]).__name__)(
            self, model_name, features)
        MODEL_CALLS[model_name+':'+type(self.estimators[model_name]).__name__] += 1
        return value

    FileExchangeModels.predict_source_probability = predict


def instrument_updates() -> None:
    from src.phase_j.live_notification_eval import NotificationExchangeOverlay
    original = NotificationExchangeOverlay.update

    @wraps(original)
    def update(self: Any, *args: Any, **kwargs: Any) -> None:
        if os.environ.get('RT_DETAIL_PROFILE') == '1' and not getattr(self, '_profile_builder', False):
            self._build_static = timed(self._build_static, 'static_features_ipc')
            self._profile_builder = True
        TIMINGS.clear()
        started = perf_counter()
        original(self, *args, **kwargs)
        elapsed = perf_counter()-started
        count = getattr(self, '_verification_count', 0)+1
        self._verification_count = count
        stamp = args[3] if len(args) > 3 else kwargs['t_sec']
        game = args[4] if len(args) > 4 else kwargs['game_idx']
        detail = dict(TIMINGS)
        detail['other'] = elapsed*MILLISECONDS-sum(detail.values())
        emit('notifications', dict(t_sec=stamp, game=game, milliseconds=elapsed*MILLISECONDS,
                                   exclusive_ms=detail))
        if count == 1 or count % SAMPLE_NOTICES == 0:
            record_state(self, stamp, game, count)

    NotificationExchangeOverlay.update = update


def record_state(overlay: Any, stamp: float, game: int, count: int) -> None:
    from src.phase_j.live_memory import retained_counts
    projection = overlay._landing_projection
    bound = getattr(projection, 'post_counter_bound', None)
    history = getattr(overlay, '_history_integrity', None)
    roots = dict(post_counter_bound=bound, history_integrity=history)
    rss = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    emit('state', dict(t_sec=stamp, game=game, notifications=count,
        configuration=configuration(overlay), model_calls=dict(MODEL_CALLS),
        retained=retained_counts(roots), rss_bytes=rss, nice=os.nice(0),
        pools={key: os.environ.get(key) for key in
               ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        post_counter_audit=list(getattr(bound, 'audit', [])),
        history_games=getattr(history, 'games', {})))


configure_comparison()
instrument_models()
instrument_breakdown()
instrument_static_builder()
instrument_updates()

if __name__ == '__main__':
    from scripts.run_live_pipeline_20260928 import main
    main()
