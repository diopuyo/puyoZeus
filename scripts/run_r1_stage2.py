"""元の描画CLIと署名を維持し、R1短区間の前後を監査する。"""
from __future__ import annotations
import argparse
from functools import wraps
import json
from pathlib import Path
import sys
from typing import Any
from scripts import collect_e34c
from scripts.run_e3_exchange_eval_20260926 import worker
from src.recognition_pipeline import RecognitionPipeline

OUT = Path('logs/r1/stage2')
FIRST, LAST = 2740.0, 2760.0


def install_trace(dest: Path) -> None:
    """既定値解決を変えず、実効設定と更新後の確定盤面だけを記録する。"""
    load, update = RecognitionPipeline.load_default, RecognitionPipeline.update
    @wraps(load)
    def configured(**kwargs: Any) -> RecognitionPipeline:
        (dest/'config.json').write_text(json.dumps(kwargs, default=str, indent=2))
        return load(**kwargs)
    @wraps(update)
    def observed(self: Any, index: int, stamp: float, frame: Any) -> Any:
        result = update(self, index, stamp, frame)
        if FIRST <= stamp <= LAST:
            sides = []
            for side in (result.p1, result.p2):
                sides.append(dict(state=side.state.name, score=side.score,
                    board=side.confirmed_board._grid.tolist() if side.confirmed_board else None))
            with (dest/'trace.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(frame=index, t=stamp, sides=sides))+'\n')
        return result
    RecognitionPipeline.load_default, RecognitionPipeline.update = configured, observed


def main() -> None:
    """OFF/ONを独立プロセスで同じ乱数・開始条件から実行する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('off', 'on'), required=True)
    options = parser.parse_args()
    dest = OUT/options.mode
    dest.mkdir(parents=True, exist_ok=True)
    collect_e34c.OUT = dest.resolve()
    args = collect_e34c.command('review')
    args[args.index('--end-sec')+1] = str(LAST)
    if options.mode == 'on':
        args.append('--placement-signal-reconcile')
    install_trace(dest)
    (dest/'command.json').write_text(json.dumps(args, indent=2))
    (dest/'trace.jsonl').write_text('')
    sys.argv = ['r1_stage2', '--worker', *args]
    worker()
    (dest/'completed.json').write_text(json.dumps(dict(mode=options.mode, completed=True)))


if __name__ == '__main__':
    main()
