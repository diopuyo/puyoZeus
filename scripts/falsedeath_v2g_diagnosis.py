"""reviewの連鎖前半を含めた段得点を、E35⁗の監査原票から全表にする。"""
from __future__ import annotations

import json
from collections import Counter

from scripts.falsedeath_replay_20261001 import OUT
from scripts.falsedeath_v2d_report import table

DEST = OUT/'v2g'
SCENE_START, SCENE_END = 2760., 2766.
MIA_GAME = 7


def extract(source: str) -> dict:
    """期限内reviewとmia7について、各連鎖の最新照合と全呼出し結果を保存する。"""
    rows = json.loads((DEST/'short'/source/'post_counter_bound_audit.json').read_text())['rows']
    selected = [r for r in rows if r['side'] == '2P' and r.get('completion_prefix_check')
        and ((source == 'review' and SCENE_START <= r['t_sec'] <= SCENE_END)
             or (source != 'review' and r['game'] == MIA_GAME))]
    latest = {}
    for row in selected:
        latest[row['chain_id']] = dict(t_sec=row['t_sec'], chain_id=row['chain_id'],
                                       check=row['completion_prefix_check'])
    return dict(calls=len(selected), first_sec=selected[0]['t_sec'] if selected else None,
        last_sec=selected[-1]['t_sec'] if selected else None,
        reasons=dict(Counter(r['completion_prefix_check']['reason'] for r in selected)),
        chains=list(latest.values()))


def render(values: dict) -> str:
    """段の飛びを合算一致として採用せず、欠測を区別して表示する。"""
    text = table('対象区間の抑止', ['記録', '呼出し', '初回秒', '最終秒', '理由'],
        [[name, *[row[k] for k in ('calls', 'first_sec', 'last_sec', 'reasons')]] for name, row in values.items()])
    rows = []
    for group in values['review']['chains']:
        for idx, candidate in enumerate(group['check']['candidates']):
            for row in candidate['comparisons']:
                observed = row['observed_increments']
                rows.append([group['chain_id'], idx, row['previous_count'], row['count'],
                    row['predicted_increment'], observed.get('formula'), observed.get('display'),
                    row['contiguous'], row['offsets']])
    text += table('review期限内・観測済み全段の照合',
        ['連鎖ID', '候補', '前段', '段', '予測増分', '式増分', '表示増分', '隣接段', '累積基準差'], rows)
    text += ('2→3段目は予測640点、式・表示900点で不一致。後半の一定260点差は、この段で生じている。'
             '6段目の実読欠測による5→7段の合算一致も、各段一致の証拠には使わない。\n\n')
    text += 'mia7の保存入力は式の実読欠測。映像で確認済みの段番号を判定へ注入していない。\n'
    return text


def main() -> None:
    """原票を変えず、機械可読の照合根拠と報告の全表を追加する。"""
    values = {source: extract(source) for source in ('mia8KCjr52g', 'review')}
    with (DEST/'STEP_EVIDENCE.json').open('x', encoding='utf-8') as stream:
        json.dump(values, stream, ensure_ascii=False, indent=1)
    with (DEST/'REPORT.md').open('a', encoding='utf-8') as stream:
        stream.write(render(values))


if __name__ == '__main__':
    main()
