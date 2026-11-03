"""短区間の照合値を監査原票から抽出し、停止理由を再現可能にする。"""
from __future__ import annotations

import json
from collections import Counter

from scripts.falsedeath_replay_20261001 import OUT
from scripts.falsedeath_v2d_report import table

DEST = OUT/'v2f'
SCENE_BEGIN, SCENE_DEADLINE = 2760., 2766.
MIA_GAME = 7


def extract(source: str) -> dict:
    """mia7と期限内reviewだけを取り、同一の照合値を最初の出現順でまとめる。"""
    audit = json.loads((DEST/'short'/source/'post_counter_bound_audit.json').read_text())['rows']
    rows = [r for r in audit if r['side'] == '2P' and r.get('completion_prefix_check')
            and ((source == 'review' and SCENE_BEGIN <= r['t_sec'] <= SCENE_DEADLINE)
                 or (source != 'review' and r['game'] == MIA_GAME))]
    unique = {}
    for row in rows:
        check = row['completion_prefix_check']
        identity = json.dumps(check, sort_keys=True)
        item = unique.setdefault(identity, dict(first_sec=row['t_sec'], last_sec=row['t_sec'],
            calls=0, check=check))
        item['last_sec'], item['calls'] = row['t_sec'], item['calls']+1
    return dict(calls=len(rows), outcomes=dict(Counter(r['completion_prefix_check']['reason'] for r in rows)),
                changes=list(unique.values()))


def main() -> None:
    """機械可読原票と全照合値の表を保存し、レポートへ追記する。"""
    values = {s: extract(s) for s in ('mia8KCjr52g', 'review')}
    with (DEST/'PREFIX_EVIDENCE.json').open('x', encoding='utf-8') as stream:
        json.dump(values, stream, ensure_ascii=False, indent=1)
    rows = []
    for source, value in values.items():
        for row in value['changes']:
            check = row['check']
            rows.append([source, row['first_sec'], row['last_sec'], row['calls'], check['reason'],
                check['predicted_chain_count'], check['predicted_final_score'], check['predicted'],
                check['observed_count'], check['observed_formula_cumulative'], check['formula_total'],
                check['score_display_increment']])
    text = table('照合値の全変化（mia7・review期限内2P）',
        ['記録', '初回秒', '最終秒', '呼出し', '照合', '予測総段数', '予測総得点', '予測途中得点',
         '実読段数', '実読累積', '追跡器の式累積', '表示増分'], rows)
    text += 'mia7の実映像は3段目だが、保存入力の式の実読は欠測。映像の知識を再生判定へ注入しない。\n\n'
    text += 'attempt1は元予測の途中得点を保持できなかった実装の原票。保存漏れを修正して同じ条件で再実行した。\n'
    with (DEST/'REPORT.md').open('a', encoding='utf-8') as stream:
        stream.write(text)


if __name__ == '__main__':
    main()
