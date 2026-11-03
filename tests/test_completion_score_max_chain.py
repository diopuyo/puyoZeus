"""A1bの事前登録どおりの候補優先順位とCLI配線を検証する。"""
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.chain import ChainSimulator
from src.exchange_prefire_candidates import PrefireCandidates
from src.scoring import score_to_ojama
from src.exchange_hidden_row_belief import majority_statistics
from src import production_config
from tests.test_exchange_event_evaluator import write_models
from tests.test_exchange_event_static_gfe_cli import start_render


def test_max_chain_keeps_joint_candidate_and_distribution() -> None:
    """低確率の長い候補を残し、同段数では最大確率を選ぶ。"""
    options = [dict(weight=w, score=s, prefix=tuple(range(n)), board=[[s]])
               for w, s, n in ((10, 100, 1), (1, 9000, 10), (2, 8500, 10))]
    result = majority_statistics(options, 0, max_chain=True)
    assert (result['selected_score'], result['predicted_count'], result['board']) == (8500, 10, [[8500]])
    assert (result['min_score'], result['max_score'], result['mean_score']) == (100, 9000, 27000 / 13)
    assert result['selected_probability'] == 2 / 13


def test_missing_public_prediction_republishes_same_candidate() -> None:
    """復元で公開値が欠測でも、有効候補の同じ得点・段数・盤面で送り量を返す。"""
    engine = PrefireCandidates(ChainSimulator())
    engine.coherent = engine.max_chain = True
    chain = SimpleNamespace(chain_id=1, side='1P', end_signal_sec=None,
        predicted_final_score=None, predicted_chain_count=None, predicted_final_board=None)
    stored = SimpleNamespace(**vars(chain))
    options = [dict(weight=w, score=s, prefix=tuple(range(n)), board=[[s]])
               for w, s, n in ((10, 100, 1), (1, 9000, 10), (2, 8500, 10))]
    engine.entries[1] = dict(chain=stored, options=options, elapsed=0.,
        original=(None, None, None), published=None, audit={})
    sent = engine.sends([chain], 0.)
    assert sent == [float(score_to_ojama(8500, elapsed_sec=0.).ojama_count), 0.]
    assert (chain.predicted_final_score, chain.predicted_chain_count, chain.predicted_final_board) == (8500, 10, [[8500]])
    assert engine.entries[1]['published'] == (8500, 10, [[8500]])
    assert engine.entries[1]['chain'] is chain
    assert stored.predicted_final_score is None


@pytest.mark.parametrize('enabled', [False, True])
def test_max_chain_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """A1b単独フラグが候補選択と既存補完を同時に有効にする。"""
    write_models(tmp_path)
    flags = ['--exchange-event-update', '--exchange-event-model-dir', str(tmp_path),
             '--prefire-snapshot', '--hidden-row-belief']
    if enabled:
        flags.append('--completion-score-max-chain')
    overlay = start_render(monkeypatch, flags)
    assert overlay._prefire.max_chain is enabled
    assert overlay._prefire.coherent is enabled
    assert (overlay._completion_score is not None) is enabled
    # 2026-10-03 本番登録済み (A1B_ADOPTION_REASON)。
    assert '--completion-score-max-chain' in production_config.exchange_event_flags()
