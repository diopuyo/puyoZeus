"""原実装の関数をそのまま切り出し、学習・採点用importを配布経路から外す。"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = '''"""v4.9fの発火起点予測。元レビュー関数から学習・ファイル入出力依存を分離。"""
from __future__ import annotations
from copy import deepcopy
from time import perf_counter
from typing import Any
from src.exchange_count_prediction import distribution, LOW_SPEC_RATIO, COLS, simulate
from src.exchange_count_v34 import candidates, probabilities, condition
from src.exchange_count_v310 import update
from src.exchange_count_correction import RemainingCorrection
from src.exchange_correction_v310 import observation_gate
from src.exchange_count_ledger import gross, STEP_SEC, FORMULA_SCORE_DELAY_SEC
from src.prefire_confidence import candidate_summary

MS, MAX_LOW_SPEC_MS, FORMULA_GAP = 1000., 500., 2.
'''
SOURCES = {
    'predict_exchange_counts_v33': ('with_send', 'formula_step'),
    'predict_exchange_counts_v34': ('predict',),
    'replay_exchange_v310': ('point', 'record', 'sent_values', 'initialize', 'on_formula', 'on_score'),
    'prefire_v45_origins': ('snapshot', 'image_observation_needed', 'terminal_observation'),
    'prefire_v46_origins': ('snapshot',),
}


def main() -> None:
    pieces = [HEADER]
    for module, names in SOURCES.items():
        text = (ROOT/'scripts'/f'{module}.py').read_text(encoding='utf-8')
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                function = ast.get_source_segment(text, node)
                if module == 'prefire_v45_origins' and node.name == 'snapshot':
                    function = function.replace('def snapshot(', 'def base_snapshot(')
                function = function.replace('BASE_SNAPSHOT(', 'base_snapshot(')
                pieces.append(function)
    (ROOT/'src/prefire_origin_stream.py').write_text('\n\n\n'.join(pieces)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
