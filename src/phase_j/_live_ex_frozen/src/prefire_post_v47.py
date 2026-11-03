"""既定OFFのv4.7。送り候補を変えず、得点台帳との接続を修正する。"""
from __future__ import annotations
from src.phase_j._live_ex_frozen.src.prefire_post_v46 import PostV46
from src.phase_j._live_ex_frozen.src.prefire_pending_v47 import ConnectedLedger, project
from src.phase_j._live_ex_frozen.src.exchange_count_ledger import gross, STEP_SEC, FORMULA_SCORE_DELAY_SEC


def remaining_send(chain: dict | None, generated: float, first: float) -> int:
    """盤面終了と得点反映は別。既知の段の遅着得点を完走通知で捨てない。"""
    if chain is None:
        return 0
    scores = chain['best']['scores']
    if chain['finished'] and not chain.get('external'):
        verified = max((o['count'] for o in chain['observed']), default=0)
        scores = scores[:verified]
    times = [chain['started']+FORMULA_SCORE_DELAY_SEC+k*STEP_SEC for k in range(len(scores))]
    total = gross(scores, times, int(chain['carry_at_fire']), first)[0]
    # 画像・式入力があるときはv3.10の反映時刻換算をそのまま継承する。
    total = chain.get('external_sent', total)
    total += chain.get('all_clear_bonus', 0)
    reflected = generated-chain['generated_at_fire']
    return int(max(0, total-reflected))


class PostV47(PostV46):
    def _boundary(self, stamp: float, game: int) -> None:
        changed = self.state.game_idx != game
        super()._boundary(stamp, game)
        if changed:
            self.state.sl = ConnectedLedger(getattr(self, 'defer_all_clear', False), self.native)

    def projection(self) -> dict:
        if self.state.sl.core.defer_all_clear:
            for side, chain in enumerate(self.chains):
                if chain is not None:
                    chain['all_clear_bonus'] = self.state.sl.core.all_clear[side].forecast(chain['started'])
        value = super().projection()
        snap = self.state.sl_snap
        amounts, triggers, ends = [], [], []
        for i, chain in enumerate(self.chains):
            active = chain is not None and not chain['finished']
            scores = chain['best']['scores'] if active else []
            times = [chain['started']+FORMULA_SCORE_DELAY_SEC+k*STEP_SEC for k in range(len(scores))]
            first = snap['margin_origin_sec']
            first = self.state.first_move_t() if first is None else first
            first = self.state.game_start_t if first is None else first
            amounts.append(remaining_send(chain, snap[f'generated_p{i+1}'], first))
            triggers.append(chain['started'] if active else None)
            ends.append(max(self.state.t, times[-1]) if times else None)
        projection = project(self.state.sl.core, amounts, self.state.t, triggers, ends)
        value.update(projection, remaining=amounts, current_pending=projection['pending_observed'],
            now=projection['pending_observed'][1]-projection['pending_observed'][0],
            post=projection['pending'][1]-projection['pending'][0],
            rate=snap['rate'], margin_origin_sec=snap['margin_origin_sec'],
            predicted_scores=[c['best']['scores'] if c else [] for c in self.chains],
            generated_at_fire=[c['generated_at_fire'] if c else None for c in self.chains])
        return value
