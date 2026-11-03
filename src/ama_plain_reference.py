"""方式Fの1回読み。盤面と可視3組のみをamaへ渡す（予告・相手情報なし）。"""
from __future__ import annotations
from collections import OrderedDict
import json
from pathlib import Path
from time import perf_counter
from src.ama_evaluation_v3_reference import Reference as BaseReference

BINARY = Path('logs/ama_plain/reference')
WIDTH, DEPTH, BRANCHES = 250, 16, 6
CACHE_LIMIT = 512


def request_key(board: list, visible: list, colors: tuple) -> tuple:
    if len(visible) != 3 or len(board) != 78:
        raise ValueError('可視3組と78セルが必要')
    mapping = {0: 0, 9: 9, **{c: i+1 for i, c in enumerate(colors)}}
    req = dict(board=[mapping[c] for c in board],
        visible=[[mapping[c] for c in pair] for pair in visible], op='plain',
        branches=BRANCHES, width=WIDTH, depth=DEPTH)
    return json.dumps(req, separators=(',', ':')), mapping


class Reference(BaseReference):
    def __init__(self, binary: Path = BINARY, cache: bool = True, process: object | None = None) -> None:
        super().__init__(binary, process=process)
        self.cache = OrderedDict()
        self.use_cache = cache
        self.expanded_nodes = {k: 0 for k in self.metrics}

    def think(self, board: list, visible: list, colors: tuple) -> dict:
        key, mapping = request_key(board, visible, colors)
        began = perf_counter()
        hit = self.use_cache and key in self.cache
        if hit:
            raw = self.cache[key]
            self.events['reused_thinks'] += 1
        else:
            self.process.stdin.write(key+'\n')
            self.process.stdin.flush()
            raw = json.loads(self.process.stdout.readline())
            if 'error' in raw:
                self.fail(raw['error'])
            self.calls += 1
            self.cpu_sec += raw['cpu_sec']
            self.expanded_nodes[self.stage] += raw['expanded_nodes']
            if self.use_cache:
                self.cache[key] = raw
                if len(self.cache) > CACHE_LIMIT:
                    self.cache.popitem(last=False)
        result = json.loads(json.dumps(raw))
        reverse = {v: k for k, v in mapping.items()}
        if 'initial' in result:
            result['initial']['board'] = [reverse[c] for c in result['initial']['board']]
        for candidate in result['candidates']:
            for step in candidate['steps']:
                step['board'] = [reverse[c] for c in step['board']]
        result.update(wall_sec=perf_counter()-began, reused=hit)
        return result

    def prime(self, board: list, visible: list, colors: tuple, plan: dict) -> None:
        """較正時と同じ盤・3組の保存計画を初手の読取りへ戻す。送り等の会計値は使わない。"""
        key, mapping = request_key(board, visible, colors)
        candidates = []
        if not plan['no_safe_move']:
            steps = json.loads(json.dumps(plan['steps']))
            for step in steps:
                step['board'] = [mapping[c] for c in step['board']]
            candidates = [dict(x=plan['placement'][0], r=plan['placement'][1],
                score=plan['ama_score'], root_score=sum(steps[0]['scores']), steps=steps,
                representative_branch=plan['representative_branch'], branches=plan['branches'])]
        self.cache[key] = dict(candidates=candidates, cpu_sec=0., expanded_nodes=0)
