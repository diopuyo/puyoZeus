"""E35を死亡専用入力と予測層へ接続し、証明できない場合は従来探索へ戻す。"""
from __future__ import annotations

from time import perf_counter
from typing import Any

from src.board import Board
from src.match_color_evidence import MatchColorEvidence
from src.post_counter_death_bound import prove_post_counter
from src.scoring import compute_effective_rate

CACHE_LIMIT = 2048


class PostCounterDeathBound:
    """盤面確率は変更せず、全候補の死亡証明と所要時間だけを保存する。"""

    def __init__(self, early_exit: bool = False) -> None:
        # 既定OFF (2026-09-30): 全候補が死ぬ場合だけ「死亡」と判定するので、最初に死なない候補が
        # 見つかった時点で判定は確定する。ON では以降の候補の証明を省く (判定は同一、監査の proofs だけ短くなる)。
        self.early_exit = early_exit
        self.colors = [MatchColorEvidence(), MatchColorEvidence()]
        self.cache: dict = {}
        self.board_cache: dict = {}
        self.audit: list[dict] = []
        self.game: int | None = None
        self.queues: tuple = ((), ())
        self.revision = 0

    def observe(self, result: Any, game: int) -> None:
        """試合境界で色情報を初期化する。未来の観測は参照しない。"""
        if self.game != game:
            self.game = game
            self.colors = [MatchColorEvidence(), MatchColorEvidence()]
            self.cache.clear()
            self.board_cache.clear()
        for evidence, side in zip(self.colors, (result.p1, result.p2)):
            evidence.observe(side)
        queues = tuple(tuple(int(c) for c in (*(getattr(s, 'next_pair', None) or (0, 0)),
                                              *(getattr(s, 'dnext_pair', None) or (0, 0))))
                       for s in (result.p1, result.p2))
        if queues != self.queues:
            self.queues = queues
            self.revision += 1

    def prove(self, projection: Any, overlay: Any, idx: int, latest: tuple,
              incoming: int, hands: int, context: dict, stamp: float) -> dict:
        """複数候補は全て判定する。単一予測の盤面欠測は補完しない。"""
        chain = overlay.tracker.latest_chain(f'{idx+1}P')
        hidden = context['hidden'][idx]
        if chain is None or not context['certain'][idx] or context['credit'][idx]:
            return dict(dead=False, reason='missing_post_counter_completion')
        if not projection._known_budget(overlay.tracker, 1-idx, stamp) or not context['verified'][idx]:
            return dict(dead=False, reason='unverified_attack_or_budget')
        if projection._chaining(overlay.tracker, idx) and hidden is None and chain.predicted_final_board is None:
            return dict(dead=False, reason='missing_majority_board')
        boards = ([Board.from_list(v['board']) for v in hidden['options']] if hidden
                  else [context['replies'][idx]])
        palette = self.colors[idx].active()
        queue = self.queues[idx]
        elapsed = overlay.tracker._score_elapsed
        key = (tuple(b._grid.tobytes() for b in boards), incoming, queue, hands, palette,
               compute_effective_rate(elapsed))
        cached = key in self.cache
        started = perf_counter()
        if not cached:
            if len(self.cache) >= CACHE_LIMIT:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = self.proofs_for(boards, queue, incoming, hands, elapsed, palette)
        proofs = self.cache[key]
        dead = bool(proofs) and all(p['dead'] for p in proofs)
        value = dict(dead=dead, reason='post_counter_upper_bound' if dead else 'fallback_search',
                     candidates=len(boards), proofs=proofs, prediction_included=True)
        self.audit.append(dict(game=overlay._game, side=f'{idx+1}P', chain_id=chain.chain_id,
            t_sec=stamp, incoming=incoming, hands=hands, palette=palette, queue=queue,
            completion_score=hidden['score'] if hidden else chain.predicted_final_score,
            elapsed_sec=perf_counter()-started, cached=cached, **value))
        return value

    def proofs_for(self, boards: list[Board], queue: tuple, incoming: int, hands: int,
                   elapsed: float, palette: tuple) -> list[dict]:
        """候補一覧の変化を跨いで同一入力の全証明を共有する（試合内・有限）。"""
        inputs = (queue, incoming, hands, compute_effective_rate(elapsed), palette)
        proofs = []
        for board in boards:
            key = (board._grid.tobytes(), *inputs)
            if key not in self.board_cache:
                if len(self.board_cache) >= CACHE_LIMIT:
                    self.board_cache.pop(next(iter(self.board_cache)))
                self.board_cache[key] = prove_post_counter(board, queue, incoming, hands, elapsed, palette)
            proofs.append(self.board_cache[key])
            if self.early_exit and not proofs[-1]['dead']:
                break
        return proofs

    def evaluate(self, projection: Any, overlay: Any, latest: tuple, hands: tuple,
                 stamp: float, value: dict, context: dict | None) -> None:
        """新規証明だけを回避不能死へ追加し、既存の予測込み表示を利用する。"""
        if context is None:
            return
        rows = []
        for idx, incoming in enumerate(context['incoming']):
            side = f'{idx+1}P'
            if incoming <= 0 or side in value['dead_sides']:
                continue
            result = self.prove(projection, overlay, idx, latest, incoming, hands[idx], context, stamp)
            rows.append(dict(side=side, **result))
            if result['dead']:
                value['dead_sides'].append(side)
        value['post_counter_bound'] = rows
