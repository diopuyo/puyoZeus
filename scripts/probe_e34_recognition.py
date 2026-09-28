"""元の描画コマンドでもW48b補完の認識入力が一致するか短区間で確認する。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from scripts.enrich_e34_origins import attach
from src.exchange_event_record import ExchangeEventRecorder, read_records
from src.recognition_pipeline import RecognitionPipeline

SOURCE = 'q_7gc4TgFig'
SECONDS = 3
OUT = Path('logs/e34/full_recognition_probe')


def main() -> None:
    """副作用を含む元の描画経路で、保存認識との一致だけを照合する。"""
    from scripts.run_e3_exchange_eval_20260926 import worker, save_json
    OUT.mkdir(parents=True, exist_ok=True)
    source = Path('logs/e8/renders')/SOURCE/'on'
    command = json.loads((source/'status.json').read_text())['command']
    stream = read_records(Path('logs/e31/records')/f'{SOURCE}.jsonl.gz')
    saved = (r for r in stream if r['kind'] == 'update')
    original, load = ExchangeEventRecorder.update, RecognitionPipeline.load_default
    frames = 0
    def capture(self: Any, result: Any, *args: Any) -> None:
        nonlocal frames
        row = next(saved)
        assert row['args'][3] == args[2]
        attach(row['args'][0], result, args[2])
        original(self, result, *args)
        frames += 1
    def configured(**kwargs: Any) -> RecognitionPipeline:
        return load(**dict(kwargs, enable_landing_chain_record_hold=True, enable_chain_active_record_hold=True))
    ExchangeEventRecorder.update, RecognitionPipeline.load_default = capture, configured
    sys.argv = ['probe', *[v.replace(str(source.resolve()), str(OUT.resolve())) for v in command[3:]]]
    sys.argv[sys.argv.index('--max-sec')+1] = str(SECONDS)
    sys.argv.append('--no-render')
    try:
        worker()
        save_json(OUT/'result.json', dict(state='matched', frames=frames))
    except ValueError as error:
        save_json(OUT/'result.json', dict(state='mismatch', frames=frames, error=str(error)))
        raise


if __name__ == '__main__':
    main()
