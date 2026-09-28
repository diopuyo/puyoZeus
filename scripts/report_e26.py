"""E26の切り分け表・途中予測件数・固定基準の合否をまとめる。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e26')
LABELS = dict(E22='E22基準', ledger='交換独立台帳', color='色得点・静穏条件',
              multi='複数着弾', safety='landing-state-safety単独', recovery='起点補完単独')


def ablation_table(ablation: dict) -> list[str]:
    """各行を同じ固定母数で表示し、実際の誤発火母数も残す。"""
    rows, lines = [], ['|条件|q log loss|zenchi一致|誤発火|', '|---|---:|---:|---:|']
    for name, result in ablation['results'].items():
        q, z, death = result['q'], result['zenchi'], result['deaths']
        row = dict(condition=name, q_log_loss=q['log_loss'], q_frames=q['frames'],
            q_matches=q['matches'], zenchi_hits=z['hits'], zenchi_frames=z['frames'],
            false=death['false'], firings=death['total'], unlabelled=death['unlabelled'])
        rows.append(row)
        lines.append(f"|{LABELS[name]}|{q['log_loss']:.9f}|{z['hits']}/{z['frames']} "
                     f"({100*z['agreement']:.4f}%)|{death['false']}/{death['total']}|")
    with (OUT/'ABLATION.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return lines


def main() -> None:
    """全条件を満たさない時はE22候補を維持し、動画を生成済みと書かない。"""
    ablation = json.loads((OUT/'ABLATION.json').read_text())
    metrics = json.loads((OUT/'on/METRICS.json').read_text())
    counts = json.loads((OUT/'MIDCHAIN_COUNTS.json').read_text())
    base = ablation['results']['E22']['q']['log_loss']
    worsened = [name for name, r in ablation['results'].items() if r['q']['log_loss'] > base]
    lines = ['# E26検収結果', '', *ablation_table(ablation), '',
        'q母数は全条件6,526行・4試合。zenchiは8,333行。reviewは誤発火母数へ重複算入しない。',
        'q悪化条件: '+', '.join(LABELS[n] for n in worsened),
        '組合せ: '+', '.join(ablation['selected'])+' + midchain_completion', '',
        '## 途中予測の件数（記録再生3動画）', '',
        json.dumps(counts, ensure_ascii=False, indent=2), '', '## 固定基準の判定', '',
        json.dumps(metrics, ensure_ascii=False, indent=2), '',
        '全条件合格。レビュー動画生成対象。' if metrics['candidate'] else
        '不合格。採用候補はE22のまま。レビュー動画は生成しない。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save_json(OUT/'SUMMARY.json', dict(q_worsened=worsened, selected=ablation['selected'],
        midchain=counts['three_videos'], metrics=metrics, video_requested_if_pass=True))


if __name__ == '__main__':
    main()
