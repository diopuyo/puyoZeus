"""5c7c6333/scripts/prefire_v410_obs_ledger.pyの得点窓（保存パス依存を除去）。"""
from __future__ import annotations

from ._live_ex_frozen.src.b3_score_ledger import ScoreLedgerV2, SCORE_PERSIST_SEC


class ObsLedger(ScoreLedgerV2):
    """正式OBS版と同じ採用処理。C2側の台帳には触れない。"""

    def _adopt_score(self, i: int, raw: int | None, t_now: float) -> list[tuple[int, float]]:
        st = self.st[i]
        if raw is None:
            st.raw_win.clear()
            return []
        window = SCORE_PERSIST_SEC
        if st.raw_win:
            gap = t_now - st.raw_win[-1][0]
            if gap <= 0:
                return []
            window = max(window, gap)
        st.raw_win.append((t_now, raw))
        while st.raw_win and t_now - st.raw_win[0][0] > window:
            st.raw_win.popleft()
        if len(st.raw_win) < 2:
            return []
        cand = min(v for _, v in st.raw_win)
        t_event = st.raw_win[-2][0]
        if st.adopted is None:
            return self._adopt_first(i, cand, t_event)
        if cand > st.adopted:
            delta = cand - st.adopted
            st.adopted = cand
            st.regress_since = None
            return [(delta, t_event)]
        self._track_regress(i, raw, t_now)
        return []
