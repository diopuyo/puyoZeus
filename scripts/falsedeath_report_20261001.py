"""採点完了後だけ、事前登録の判定と数値をMarkdownへ出力する。"""
from __future__ import annotations

import json
from pathlib import Path
from scripts.falsedeath_collect_20261001 import OUT, ROOT


def interval(value: dict) -> str:
    """対差と95%区間を同じ精度で示す。"""
    a, b = value['ci95']
    return f"{value['diff']:+.6f} [{a:+.6f}, {b:+.6f}]"


def delay_text(value: dict) -> str:
    """未観測は欠測数として示し、遅れゼロに置換しない。"""
    q = value['lost_delay'].get('quantiles', {})
    observed = ('なし' if not q else
        f"最小 {q['min']:.3f}秒、P25 {q['p25']:.3f}秒、中央値 {q['p50']:.3f}秒、P75 {q['p75']:.3f}秒、P95 {q['p95']:.3f}秒、最大 {q['max']:.3f}秒")
    return f"消失 {value['lost_count']}件、観測死亡までの遅れ: {observed}、観測死亡欠測 {value['missing_observed']}件"


def set_lines(name: str, value: dict) -> list[str]:
    """母数・対差・誤確定をセット別に示す。"""
    d, m = value['deaths'], value['metrics']
    return [f"### {name} ({value['games']}試合)", '',
        f"- 誤った事前確定: {d['before']['false']}/{d['before']['cases']} → {d['after']['false']}/{d['after']['cases']}。",
        f"- 誤った確定表示: {d['before']['false_frames']} → {d['after']['false_frames']}フレーム。",
        f"- M_time: {m['m_time']['prod']:.6f} → {m['m_time']['all']:.6f}、差 {interval(m['m_time']['paired'])}。",
        f"- M_agree: {m['m_agree']['prod']:.6f} → {m['m_agree']['all']:.6f}、差 {interval(m['m_agree']['paired'])}。",
        f"- 正しい事前確定: 基準 {d['correct_before']}件、保持 {d['correct_retained']}件。{delay_text(value['correct_delay'])}。", '']


def ablation_lines(results: dict) -> list[str]:
    """各一つ抜きの参考値を記載し、合否には混ぜない。"""
    lines = ['## 一つ抜きの寄与（参考・合否対象外）', '']
    for mode in ('no_d1', 'no_d2', 'no_d3', 'no_d4'):
        result = results[mode]
        numbers = []
        for name, value in result['sets'].items():
            m = value['metrics']['m_time']
            full = results['all']['sets'][name]['metrics']['m_time']['all']
            a = value['metrics']['m_agree'][mode]
            full_a = results['all']['sets'][name]['metrics']['m_agree']['all']
            numbers.append(f"{name}: 誤確定 {value['deaths']['after']['false']}件、M_time {m[mode]:.6f} (全ONとの差 {m[mode]-full:+.6f})、M_agree {a:.6f} (全ONとの差 {a-full_a:+.6f})")
        lines.append(f"- {mode}: 既存5記録の誤確定 {result['existing5']['false']}件。"+' / '.join(numbers)+'。')
    return lines+['']


def failure_lines(full: dict) -> list[str]:
    """残存事例は母数から隠さず、元の試合・側・時刻で示す。"""
    lines = ['## 残る誤確定の原票', '']
    for name, value in full['sets'].items():
        after = value['deaths']['after']
        rows = [r for r in after['rows'] if r['false']]
        lines += [f"- {name} 第{r['game']}試合 {r['side']}: 初回 {r['first_sec']:.6f}秒。" for r in rows]
        lines += [f"- {name} 第{r['game']}試合の誤確定表示: {r['count']}フレーム、初回 {r['first']:.6f}秒。"
                  for r in after['wrong_frames']]
    lines += [f"- 既存5記録 {r['source']} 第{r['game']}試合 {r['side']}: {r['first_sec']:.6f}秒、誤確定={r['false']}、未解決={r['unresolved']}。"
              for r in full['existing5']['rows']['all'] if r['false'] or r['unresolved']]
    return lines + (['なし。', ''] if len(lines) == 2 else [''])


def main() -> None:
    """全構成の採点原票がある場合だけ検証中の文書を最終結果へ置換する。"""
    results = json.loads((OUT/'RESULT.json').read_text())
    full = results['all']
    lines = ['# 誤った負け確定の根治 D1〜D4', '',
        f"**判定: {'合格' if full['passed'] else '不合格'}**。本番登録は行っていない。4フラグとも既定OFF。", '',
        '開始HEAD `ff67208`。事前登録の原文は [PREREGISTRATION.md](falsedeath_fix/PREREGISTRATION.md)。',
        '`src/production_config.py` は未変更。実画像はGitに含めない。', '', '## 条件1〜4', '']
    labels = dict(false_death='条件1 誤った負け確定0', m_time='条件2 M_time区間上端 < +0.002',
                  m_agree='条件3 M_agree区間下端 > −0.01', regression='条件4 3:00場面・第14試合')
    lines += [f"- {labels[k]}: {'PASS' if v else 'FAIL'}。" for k, v in full['gates'].items()]
    lines += ['']
    for name, value in full['sets'].items():
        lines += set_lines(name, value)
    old = full['existing5']
    lines += ['## 既存5記録・回帰・条件5', '',
        f"事前確定 {old['before']} → {old['after']}件、誤確定 {old['false']}件、未解決 {old['unresolved']}件。", '']
    lines += [f"- {name}: {delay_text(value['correct_delay'])}。" for name, value in old['sources'].items()]
    target = full['regression']
    lines += ['', f"3:00場面: {target['scene']['legacy_metric_sec']}秒 (基準 ≤ 2766.0)。実表示の初到達 {target['scene']['first_display_sec']}秒。",
              f"q第14試合の誤確定: {len(target['game14_false_times'])}件。", '',
              '消失・遅れは合否に用いていない。遅れは修正側の観測死亡時刻−本番側の初回事前確定時刻（負値は先行）。欠測は0秒にしない。全事例の側・時刻、保持した事例の遅れの分布はJSON原票に保存。', '']
    lines += ablation_lines(results) + failure_lines(full)
    test_summary = next(line for line in (OUT/'TESTS.log').read_text().splitlines() if 'passed' in line and ' in ' in line)
    lines += [f'関連テスト: **{test_summary}**。全testsの実行結果ではない。', '']
    lines += ['## 実装・再現性', '',
        '- D1 `--death-input-integrity`: 不整合を旧死亡へ反転しない。',
        '- D2 `--post-counter-input-integrity`: 不安定入力・NEXT移動を拒否。予測手数を証明上限にしない。',
        '- D3 `--score-attribution`: 数値増加で式の帰属を制限し、一致した実測で早期確定を訂正。',
        '- D4 `--falling-puyo-integrity`: STABLE復旧と着地色監視で操作中の色セルを固定しない。',
        '- 元コマンドの区間・入力動画・ウォームアップ・間引き・モデルを維持し、同じ収集器へ4フラグだけ追加。',
        '- 本番とno_d4は元記録、全ONと他の一つ抜きは再収集記録。全18入力を本番CLIで独立再生し、その出力だけを採点。収集時の補助評価イベントは採点対象外。',
        '- 試合単位の対ブートストラップ10,000回。既存採点器の種（セット1=20261001、セット2=0）を維持。',
        '- 診断6件とD4の実保存書込み盤面による局所検査は `falsedeath_fix/LOCAL.json`。第41試合は880→39,700点、実保存台帳の未相殺残量282→0個を確認。',
        '- 全数値・全事例は `falsedeath_fix/RESULT.json`、コード・実行構成は `PLAN.json`、テストは `TESTS.log`。', '']
    (ROOT/'logs/eval_set/FALSEDEATH_FIX.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
