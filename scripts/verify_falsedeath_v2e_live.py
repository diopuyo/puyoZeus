"""統合worktreeを変更せず、実ライブの評価入力でH1′開始証拠を観測する。"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

# ライブ本体は実行時cwd、候補の観測器だけはこのworktreeから読み込む。
sys.path.insert(0, str(Path.cwd()))
CANDIDATE = Path(__file__).resolve().parents[1]/'src/exchange_history_integrity.py'
SPEC = importlib.util.spec_from_file_location('candidate_h1_prime', CANDIDATE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
STREAM: Any = None


def install() -> None:
    """通常の評価を変えず、評価workerが受信した全updateを同じ候補へ渡す。"""
    from src.exchange_event_overlay import ExchangeEventOverlay
    original = ExchangeEventOverlay._observe_prefire

    def observe(self: Any, sides: tuple, stamp: float, game: int) -> None:
        global STREAM
        original(self, sides, stamp, game)
        if not hasattr(self, '_candidate_h1_prime'):
            self._candidate_h1_prime = MODULE.MatchHistoryIntegrity(boundary_score_start=True)
            self._candidate_h1_previous = None
        guard = self._candidate_h1_prime
        guard.observe(sides, game, stamp)
        evidence = guard.evidence()
        current = (game, evidence['history_complete'])
        if current == self._candidate_h1_previous:
            return
        self._candidate_h1_previous = current
        if STREAM is None:
            directory = Path(os.environ['H1_PRIME_OUTPUT'])
            directory.mkdir(parents=True, exist_ok=True)
            STREAM = (directory/f'evidence-{os.getpid()}.jsonl').open('x', encoding='utf-8')
        row = dict(game=game, t_sec=stamp, scores=[s.score for s in sides],
            states=[s.state.name for s in sides], evidence=evidence, nice=os.nice(0),
            original_h1=self._history_integrity.evidence())
        STREAM.write(json.dumps(row, ensure_ascii=False)+'\n')
        STREAM.flush()

    ExchangeEventOverlay._observe_prefire = observe


install()

if __name__ == '__main__':
    from scripts.run_live_pipeline_20260928 import main
    main()
