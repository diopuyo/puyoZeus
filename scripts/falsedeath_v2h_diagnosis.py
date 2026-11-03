"""短区間の直近窓と過去不一致を、推測せず監査原票から抜き出す。"""
from __future__ import annotations

import json

from scripts.falsedeath_replay_20261001 import OUT
from scripts.falsedeath_v2d_report import table

DEST = OUT/'v2h'
REVIEW_BEGIN, REVIEW_DEADLINE = 2750., 2766.
MIA_GAME = 7


def evidence(source: str) -> dict:
    """review初回許可・証明とmia7の照合欠測を独立して保存する。"""
    rows = json.loads((DEST/'short'/source/'post_counter_bound_audit.json').read_text())['rows']
    rows = [r for r in rows if r['side'] == '2P' and 'completion_prefix_check' in r and (
        REVIEW_BEGIN <= r['t_sec'] <= REVIEW_DEADLINE if source == 'review' else r['game'] == MIA_GAME)]
    accepted = [r for r in rows if r['completion_prefix_check']['accepted']]
    proven = [r for r in accepted if r.get('dead')]
    return dict(calls=len(rows), accepted=len(accepted), proven=len(proven),
        first=rows[0] if rows else None, last=rows[-1] if rows else None,
        first_accepted=accepted[0] if accepted else None, first_proven=proven[0] if proven else None)


def main() -> None:
    """実読された段と予測増分を再現可能なJSONで保存する。"""
    value = {source: evidence(source) for source in ('review', 'mia8KCjr52g')}
    with (DEST/'DIAGNOSIS.json').open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1)
    print(json.dumps({s: {k: r[k] for k in ('calls', 'accepted', 'proven')} for s, r in value.items()}))


def render(value: dict) -> str:
    """初回死亡証明の窓と過去比較を、観測していない表示値を補わず示す。"""
    text = table('review期限内とmia7の照合', ['記録', '呼出し', '照合許可', '死亡証明'],
        [[source, *[r[k] for k in ('calls', 'accepted', 'proven')]] for source, r in value.items()])
    first = value['review']['first_proven']
    if first is None:
        return text
    rows = []
    for candidate in first['completion_prefix_check']['candidates']:
        for scope, key in (('過去・監査のみ', 'earlier_comparisons'), ('直近・判定対象', 'comparisons')):
            for r in candidate[key]:
                rows.append([scope, r['previous_count'], r['count'], r['predicted_increment'],
                    r['observed_increments'].get('formula'), r['observed_increments'].get('display'),
                    r['contiguous'], r['offsets']])
    return text + f"初回死亡証明: {first['t_sec']}秒。\n\n" + table('review初回証明の全比較',
        ['扱い', '前段', '段', '予測増分', '式増分', '表示増分', '連続', '累積基準差'], rows)


if __name__ == '__main__':
    main()
