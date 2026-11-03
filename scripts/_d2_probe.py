"""D2専用。評価実装を変更せず実効設定と認識中間値を短区間で採取する。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import inspect
import json
from pathlib import Path
import sys
from typing import Any

from scripts.collect_e34b import command
from scripts.run_e3_exchange_eval_20260926 import save_json, worker
from src.exchange_event_record import encode
from src.board_state_machine import BoardStateMachine
from src.recognition_pipeline import RecognitionPipeline

OUT = Path('logs/d2')
SECONDS = 10
LOAD = RecognitionPipeline.load_default
SIGNATURE = inspect.signature(LOAD)
TRACE_FIELDS = ('state', 'confirmed_board', 'cnn_board', 'raw_cnn_board',
                'inferred_board', 'score', 'next_pair', 'dnext_pair',
                'landing_chain_started')
OUTPUT_FLAGS = ('--out', '--dump-timeline', '--dump-display-timeline',
                '--dump-exchange-events', '--exchange-event-record')


def arguments(source: str, dest: Path, seconds: int = SECONDS) -> list[str]:
    """新収集コマンドの処理開始位置を維持し、処理量だけ10秒にする。"""
    if not 0 < seconds <= SECONDS:
        raise ValueError('D2の処理区間は0秒超10秒以下に限定する')
    args = command(source)
    for flag in OUTPUT_FLAGS:
        index = args.index(flag)+1
        args[index] = str(dest/Path(args[index]).name)
    if '--max-sec' in args:
        args[args.index('--max-sec')+1] = str(seconds)
    else:
        start = float(args[args.index('--start-sec')+1])
        warmup = float(args[args.index('--warmup-sec')+1])
        args[args.index('--end-sec')+1] = str(start-warmup+seconds)
    return args


def normalized(value: Any) -> Any:
    """設定値をJSONへ変換する。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, frozenset, set)):
        return list(value)
    return value


def install_load(dest: Path, variant: str) -> None:
    """署名欠落を再現し、指定された差分だけを元の値へ戻す。"""
    def configured(**kwargs: Any) -> RecognitionPipeline:
        kwargs.update(enable_landing_chain_record_hold=True,
                      enable_chain_active_record_hold=True)
        if variant == 'holds_off':
            kwargs.update(enable_landing_chain_record_hold=False,
                          enable_chain_active_record_hold=False)
        if variant.startswith('restore:'):
            for key in variant.split(':', 1)[1].split(','):
                kwargs[key] = SIGNATURE.parameters[key].default
        effective = SIGNATURE.bind(**kwargs)
        effective.apply_defaults()
        save_json(dest/'config.json', {k: normalized(v) for k, v in effective.arguments.items()})
        return LOAD(**kwargs)
    if variant in ('signature', 'old'):
        configured.__signature__ = SIGNATURE
    if variant == 'old':
        def original(**kwargs: Any) -> RecognitionPipeline:
            effective = SIGNATURE.bind(**kwargs)
            effective.apply_defaults()
            save_json(dest/'config.json', {k: normalized(v) for k, v in effective.arguments.items()})
            return LOAD(**kwargs)
        original.__signature__ = SIGNATURE
        RecognitionPipeline.load_default = original
    else:
        RecognitionPipeline.load_default = configured


def install_trace(stream: Any) -> None:
    """返却値を変更せず、更新直後のCNN/HSV/確定盤面/状態を保存する。"""
    original = RecognitionPipeline.update
    reset_original = BoardStateMachine.reset
    resets: list[dict] = []
    def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        caller = sys._getframe(1)
        resets.append(dict(caller=caller.f_code.co_name, line=caller.f_lineno,
                           file=caller.f_code.co_filename, args=encode(args), kwargs=kwargs))
        return reset_original(self, *args, **kwargs)
    BoardStateMachine.reset = reset
    def update(self: Any, frame_idx: int, t_sec: float, frame: Any) -> Any:
        resets.clear()
        result = original(self, frame_idx, t_sec, frame)
        hsv = self._reader.read_both_boards_hsv(frame)
        sides = []
        for side, observed in zip((result.p1, result.p2), hsv):
            values = {k: encode(getattr(side, k, None)) for k in TRACE_FIELDS}
            values['hsv_board'] = encode(observed)
            values['drift'] = encode(asdict(side.drift))
            sides.append(values)
        internal = {key: normalized(value) for key, value in vars(self).items()
                    if key.startswith('_drift_resync_') or key in
                    ('_online_hsv_injected_colors', '_match_active_started_time')}
        stream.write(json.dumps(dict(t_sec=t_sec, frame_idx=frame_idx, sides=sides,
                                    internal=internal, resets=resets),
                               ensure_ascii=False, separators=(',', ':'))+'\n')
        return result
    RecognitionPipeline.update = update


def main() -> None:
    """一条件を独立プロセスで実行する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='q_7gc4TgFig')
    parser.add_argument('--variant', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seconds', type=int, default=SECONDS)
    options = parser.parse_args()
    dest = OUT/options.name
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'complete.json').unlink(missing_ok=True)
    args = arguments(options.source, dest, options.seconds)
    save_json(dest/'command.json', dict(source=options.source, variant=options.variant, args=args))
    install_load(dest, options.variant)
    with gzip.open(dest/'trace.jsonl.gz', 'wt', encoding='utf-8') as stream:
        install_trace(stream)
        sys.argv = ['d2', '--worker', *args]
        worker()
    save_json(dest/'complete.json', dict(state='completed'))


if __name__ == '__main__':
    main()
