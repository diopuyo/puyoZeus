"""D5の誤確定入力と、同じ単発死亡経路の全記録を保存する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from src.exchange_event_record import encode, read_records
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/d5')
SOURCES = ('review', 'q_7gc4TgFig', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi')
FALSE_SEC = 883.9666666666667
LONG_LEAD_SEC = 10.0


def directory(root: Path, source: str) -> Path:
    """既存検収の保存配置を返す。"""
    return root/(source if source in ('review', 'zenchi') else f'renders/{source}/on')


def read_events(source: str) -> list[dict]:
    """確定と保持を分けるため全予測行を読む。"""
    path = directory(Path('logs/e35/on'), source)/'events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit() -> None:
    """初回の単発死亡と、後続の応手・確定解除を照合する。"""
    cases = json.loads(Path('logs/e35/DEATH_AUDIT_ALL.json').read_text())['rows']
    output = []
    for source in SOURCES:
        events = read_events(source)
        for row in (r for r in cases if r['source'] == source):
            values = [v for e in events if e['game_idx'] == row['game'] for v in e['values']]
            first = next(v for v in values if v['source'] == 'unavoidable_death'
                         and abs(v['t_sec']-row['first_sec']) < 1e-6)
            idx = int(row['side'] == '2P')
            route = ('single' if first.get('multi_landing', [{}, {}])[idx].get('reason') == 'already_dead'
                     and not any(p.get('dead') and p['side'] == row['side']
                                 for p in first.get('post_counter_bound', [])) else 'other')
            chains = [dict(c) for e in events if e['game_idx'] == row['game'] for c in e['chains']
                      if c['side'] == row['side'] and c['trigger_sec'] > row['first_sec']]
            later = next((v for v in values if v['t_sec'] > row['first_sec']
                          and v['source'] == 'S3_landing' and row['side'] not in v['dead_sides']), None)
            output.append(dict(row, route=route, first=first, later_chains=chains,
                               long_lead=row['lead_sec'] is not None and row['lead_sec'] > LONG_LEAD_SEC,
                               released=later))
    save_json(OUT/'BASELINE_PATH_AUDIT.json', output)
    print([(r['source'], r['game'], r['side'], r['route'], r['lead_sec'], len(r['later_chains']))
           for r in output], flush=True)


def inputs() -> None:
    """誤確定の前後は元の全入力を保存し、STABLE履歴も切り出す。"""
    rows, latest = [], [None, None]
    for item in read_records(Path('logs/e31/records/q_7gc4TgFig.jsonl.gz')):
        if item['kind'] != 'update':
            continue
        args = item['args']
        result, _, _, stamp, *_ = args
        for idx, side in enumerate((result.p1, result.p2)):
            if side.state.name == 'STABLE' and side.confirmed_board is not None:
                latest[idx] = dict(stamp=stamp, side=encode(side))
        if 883.8 <= stamp <= 884.1 or 885 <= stamp <= 891 and result.p1.chain_event:
            rows.append(dict(stamp=stamp, input=encode(args), latest=list(latest)))
    save_json(OUT/'FALSE_INPUTS.json', rows)


def images() -> None:
    """実画面を変更せずPNGで保存する。"""
    metadata = json.loads(Path('logs/e31/records/q_7gc4TgFig.jsonl.json').read_text())
    cap = cv2.VideoCapture(metadata['video'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps > 0
    manifest = []
    for stamp in (FALSE_SEC, 885.5, 888.5, 894., 899.6333333333333):
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(stamp*fps))
        ok, frame = cap.read()
        assert ok
        path = OUT/f'q_game14_{stamp:.3f}.png'
        cv2.imwrite(str(path), frame)
        manifest.append(dict(t_sec=stamp, path=str(path), video=metadata['video']))
    cap.release()
    save_json(OUT/'IMAGES.json', manifest)


def main() -> None:
    """軽量な原票監査と証拠画像保存を順に実行する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    audit()
    inputs()
    images()


if __name__ == '__main__':
    main()
