"""元Witnessが履歴snapshotを返す責務であり現clock照合は別であることを確認。"""
from contextlib import ExitStack
from types import SimpleNamespace
from test_journal_witness import recorder, complete
import journal_witness as W


def test_original_pair_remains_readable_after_history_advances() -> None:
    journal = recorder()
    journal.history = SimpleNamespace(frame=100, time_sec=100 / 60)
    with ExitStack() as stack:
        witness = W.install(stack, journal)
        complete(journal, '1P', 0)
        complete(journal, '2P', 1)
        before, count = journal.stream.getvalue(), journal.count
        journal.history.frame, journal.history.time_sec = 102, 102 / 60
        assert all(row['frame_idx'] == 100 for row in witness.pair(100))
        assert journal.count == count and journal.stream.getvalue() == before
