"""ama想定の勝率を別枠へ公開する。保持値を新しい評価として扱わない。"""
from __future__ import annotations
import json
from pathlib import Path
from queue import Full
from time import perf_counter
from .live_ama_async import AsyncAma
from src.ama_plain_v71_display import Display

ASSETS = Path('models/ama_plain')


class LiveAma:
    def __init__(self, output: Path) -> None:
        self.worker = AsyncAma((ASSETS/'ama_reference.exe', ASSETS/'model.json', ASSETS/'seed_table_cache.npz'), output)
        self.display = Display(enabled=True)
        self.identity = None
        self.states = []
        self.failure = None
        self.completed = None
        self.reported = None
        self.input_status = None
        from .live_obs_audit import open_audit
        self.audit = open_audit(output/'ama_display.jsonl', 'x')
        self.audit.write(json.dumps(dict(kind='worker',pid=self.worker.process.pid))+'\n')

    def update(self, frame: dict, generation: int) -> None:
        identity = (generation, frame['game_idx'])
        if identity != self.identity:
            self.identity, self.display = identity, Display(enabled=True)
            self.completed, self.reported = None, None
            self.input_status = None
        self.receive()
        packet = frame.get('_ama')
        if packet is None or self.failure:
            return
        self.states = [r['state']['state'] for r in packet['raw']]
        try:
            self.worker.submit(dict(packet, generation=generation, id=frame['frame']))
        except Full:
            self.failure = 'ama観測待ち上限・再起動が必要'

    def receive(self) -> None:
        for reply in self.worker.poll():
            if reply.get('fatal'):
                self.failure = reply['fatal']
                continue
            if (reply.get('generation'),reply.get('game')) == self.identity and reply.get('input_status'):
                self.input_status = reply['input_status']
            for job in reply.get('results', []):
                if job.get('fatal'):
                    self.failure = job['fatal']
                elif (job['generation'], job['game']) == self.identity:
                    self.display.observe(job['result'])
                    if 'value' in job['result']:
                        self.completed = job

    def decorate(self, row: dict) -> None:
        self.receive()
        value = self.display.at(row['t_sec'], self.states)
        p = value['probability'] if row.get('match_active', True) else None
        plans = value['result']['value']['plans'] if value['result'] else []
        missing = self.input_status.get('missing') if self.input_status else value['missing_reason']
        pending = self.input_status and self.input_status['input_sec'] > (value['attempted_sec'] or float('-inf'))
        reason = missing or ('計算完了待ち' if pending and self.states != ['CHAIN','CHAIN'] else value['hold_reason'])
        row['ama_plain'] = dict(label='ama 想定の勝率', p1=p, p2=None if p is None else 1-p,
            input_sec=value['input_sec'], age_sec=value['hold_sec'],
            reason=self.failure or reason, missing_reason=missing,
            sides=[dict(no_safe_move=x['no_safe_move'], no_main=x['h'] is None,
                        terminal=x['terminal']) for x in plans])
        if self.completed and self.reported != value['input_sec']:
            self.reported = value['input_sec']
            self.audit.write(json.dumps(dict(input_sec=self.reported, displayed_at=perf_counter(),
                game=self.identity, frame=self.completed['frame'],
                completed_input_sec=self.completed['result']['input_sec'],
                requested_at=self.completed['requested_at'], completed_at=self.completed['completed_at'],
                received_at=self.completed.get('received_at'), collected_at=self.completed.get('collected_at'),
                request_to_display_ms=(perf_counter()-self.completed['requested_at'])*1000,
                cpu_sec=self.completed['result']['cpu_sec']))+'\n')

    def close(self) -> None:
        self.worker.close()
        self.audit.close()
