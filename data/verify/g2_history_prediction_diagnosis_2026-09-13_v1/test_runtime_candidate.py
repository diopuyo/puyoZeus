"""元factory/実Session型を使い、候補consumeの選択と保存/終了/解除を一括検査。"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
from typing import Any

ROOT = Path(__file__).resolve().parent
ASYNC = ROOT.parent / 'g2_async_projected_evaluation_2026-09-13_v1'
PUB = ROOT.parent / 'g2_belief_live_publication_2026-09-11_v1'
sys.path[:0] = [str(ASYNC), str(PUB)]
import test_runtime_patch_install as T
import session_runtime_binding_candidate as B


def test_original_factory_full_saved_rows_and_planned_outer_close(monkeypatch: Any, tmp_path: Path) -> None:
    """Mode/pipe/J初期状態は既存人工fixture、履歴rowと予測演算は原A28保存を使う。"""
    monkeypatch.setattr(T, 'B', B)
    original_setup, original_install = T.setup, B.install
    seen: list[int] = []
    def setup() -> tuple:
        journal, pipe, values = original_setup()
        journal.history.frame, journal.history.time_sec = 35160, 586.0
        return journal, pipe, values
    def prime(capture: Any, frame: int) -> None:
        assert frame == 35160 and capture.last_frame == -1
        assert Path(type(capture).__module__ and sys.modules[type(capture).__module__].__file__).resolve() == ROOT / 'journal_origin_capture_candidate.py'
        count = 0
        source = ROOT.parent / 'video38_early_origin_candidate_v28/EARLY_ORIGIN_HISTORY.jsonl'
        with source.open() as stream:
            for line in stream:
                for item in json.loads(line)['rows']:
                    capture.consume(item['row'], available_frame=frame)
                    count += 1
        assert count == 6110 and capture.ledger.snapshot(capture.handles['2P']).origin_prediction_revision == 1
        seen.append(count)
    def install(stack: Any, bootstrap: Any, replace: Any, **kwargs: Any) -> dict:
        return original_install(stack, bootstrap, replace, history_getter=lambda: N(prime=prime), **kwargs)
    monkeypatch.setattr(T, 'setup', setup)
    monkeypatch.setattr(B, 'install', install)
    T.test_original_runtime_patch_install_and_class_guard(monkeypatch, tmp_path)
    assert seen == [6110]
    rows = [json.loads(line) for line in (tmp_path / 'PROJECTED_ORIGIN_CAPTURE.jsonl').read_text().splitlines()]
    assert len([row for row in rows if row.get('kind') == 'origin_reprojection_hold']) == 4
    assert not any(alias in sys.modules for alias in B.ALIASES)
