"""probability_finishの二段目原boundaryを同じB差替で実行。人工公開票CPU。"""
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace as N
import test_terminal_publication_boundary as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'g2_empty_tail_reset_live_adapter_2026-09-11_v1'))
import live_boundary_v2 as ORIGINAL


def test_original_second_check_accepts_verified_terminal_tail() -> None:
    mode, consumer, history, journal, recovery, tracking, args = F.fixture()
    old_scope = {'synthetic_owner':True}
    history[0]['decision']['history_state'] = {'scope':old_scope}
    first = F.C.check(F.P, mode, consumer, history, journal, recovery, tracking, **args)
    # 原prepared:44-47と同じclone。元B.checkを再実行せず、検査済み境界を渡す。
    check = FunctionType(ORIGINAL.check.__code__, dict(ORIGINAL.check.__globals__,
        B=N(check=lambda *a:first | {'prior_issued_frames':first['issued_frames']})))
    second = check(consumer, history, journal, recovery, old_owner_scope=old_scope)
    assert second['prior_publication_join_verified'] and second['terminal_end'] == 8
    assert second['terminal_publication_updates'] == 2
