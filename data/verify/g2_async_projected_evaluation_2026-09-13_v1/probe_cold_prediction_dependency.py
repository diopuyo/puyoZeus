"""元A23のcold環境で私有台帳と凍結物理を接続する。人工盤面、動画更新なし。"""
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any
import session_runtime_binding as B

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.parent / 'g2_split_tail_runtime_2026-09-13_v23'
FROZEN = ROOT.parents[2] / '.runtime_snapshots/event_first30_observed_context_v5_2026-08-30'


def exercise(stack: Any, loader: Any, physics: tuple | None = None) -> dict:
    from src.board import Board
    from src.chain import ChainSimulator
    if physics is not None: Board, ChainSimulator = physics
    assert Path(sys.modules['src.board'].__file__).resolve() == FROZEN / 'src/board.py'
    assert Path(sys.modules['src.chain'].__file__).resolve() == FROZEN / 'src/chain.py'
    before = {key: sys.modules.get(key) for key in ('src.chain_commit_candidate_v1', 'src.chain_prediction_ledger_v1')}
    load = B.owned_loader(stack, loader.load)
    commit = load('_async_live_commit_types', B.REPO / 'src/chain_commit_candidate_v1.py')
    module = load('_async_live_prediction_ledger', B.REPO / 'src/chain_prediction_ledger_v1.py',
                  {'src.chain_commit_candidate_v1': commit})
    grid = [[0] * 6 for _ in range(13)]
    grid[-1][:4] = [1] * 4
    board = Board.from_dict({'grid': grid})
    event = N(before_board=board, trigger_sec=0.0, end_sec=0.0, chain_count=1,
              total_score=40, mechanism='landing', score_estimated=False)
    generation, ledger = module.ChainGeneration('1P', 0, 0), module.ChainPredictionLedger()
    point = dict(frame_idx=0, time_sec=0.0)
    handle = ledger.open_landing_provisional(generation=generation, **point,
        origin_before_board=board, landing_event=event, capture_source='artificial_cold_dependency_no_video')
    result = ChainSimulator(exclude_hidden_row_from_pop=True).simulate(board)
    prediction = ledger.add_prediction(handle, generation=generation, **point,
        episode_revision=1, input_board=board, result=result)
    assert prediction.is_origin_reference and prediction.chain_count == 1
    assert prediction.calculated_total_score == 40
    assert all(sys.modules.get(key) is value for key, value in before.items())
    return dict(board_source=sys.modules['src.board'].__file__, chain_source=sys.modules['src.chain'].__file__,
        ledger_source=module.__file__, private_prediction=asdict(prediction), source_namespace_unchanged=True,
        artificial_board=True, video_updates=0, live_hook_verified=False, quality_gate_clear=False)


def main() -> None:
    sys.path.insert(0, str(RUNTIME))
    import owned_adapter as A
    try:
        with ExitStack() as stack:
            selected = A.configured(stack)
            adapter = A.configured.__globals__['A'].A.A.V4
            owner = sys.modules[adapter.OWNED_ALIAS]
            with selected.__globals__['S'].configured():
                with ExitStack() as inner:
                    result = exercise(inner, owner.bootstrap())
                assert not any(alias in sys.modules for alias in B.ALIASES)
        result.update(configured_context_exited=True, private_aliases_released=True)
    except BaseException as error:
        result = dict(error=repr(error), quality_gate_clear=False, video_updates=0)
        raise
    finally:
        with (ROOT / 'COLD_PREDICTION_DEPENDENCY_v1.json').open('x') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
