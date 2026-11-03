"""E23の固定評価・新規死亡の原票・レビュー納品を集約する。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e23')
DOCUMENT = Path('docs/E23_MULTI_LANDING_2026-09-28.md')
MARKER = '## 最終結果'


def case_lines(cases: list[dict]) -> list[str]:
    """6ケースを、省略せず死・生存の観測時刻付きで示す。"""
    lines = ['', '|動画|試合／交換|側|区分|初判定秒|実結果|死亡観測秒|',
             '|---|---:|---|---|---:|---|---:|']
    for row in cases:
        kind = '新規' if row['change'] == 'new' else '前倒し'
        outcome = '生存（誤り）' if row['error'] else '死亡' if row['actually_died'] else '不明'
        stamp = row['observed_death_sec']
        death = '—' if stamp is None else f'{stamp:.4f}'
        lines.append(f"|{row['source']}|{row['game_idx']}／{row['exchange_id']}|{row['side']}|"
                     f"{kind}|{row['first_sec']:.4f}|{outcome}|{death}|")
    return lines


def main() -> None:
    """未丸め値の合否と実母数を記録し、不合格を候補扱いしない。"""
    baseline = json.loads(Path('logs/e22/on/METRICS.json').read_text())
    result = json.loads((OUT/'on/METRICS.json').read_text())
    cases = json.loads((OUT/'NEW_DEATH_CASES.json').read_text())
    complete = Path('logs/review_zenchi_g41_43_e23/complete.json')
    video = json.loads(complete.read_text()) if complete.exists() else None
    save_json(OUT/'METRICS.json', dict(baseline=baseline, result=result, new_cases=cases, video=video))
    lines = [MARKER, '', '|指標|E22|E23|合否|', '|---|---:|---:|---|']
    for label, key, metric in [('q log loss（6,526行）', 'q', 'log_loss'),
                               ('zenchi一致率（8,333行）', 'zenchi', 'agreement')]:
        lines.append(f"|{label}|{baseline[key][metric]:.8f}|{result[key][metric]:.8f}|{'合格' if result['gates'][key] else '不合格'}|")
    old, new = baseline['deaths'], result['deaths']
    lines.append(f"|誤発火／実発火|{old['false']}/{old['total']}|{new['false']}/{new['total']}|{'合格' if result['gates']['deaths'] else '不合格'}|")
    scene = result['scene']
    lines.append(f"|2P≤5%の初時刻|{scene['baseline_first_sec']}|{scene['first_sec']}|{'合格' if result['gates']['scene'] else '不合格'}|")
    lines += ['', '**固定評価合格・採用候補（既定OFF）**。' if result['candidate'] else
              '**固定評価不合格**。E22候補を維持し、レビュー動画は生成しない。', '',
              '3動画だけの新規・前倒し原票: `logs/e23/NEW_DEATH_CASES.csv`（JSONも併記）。',
              f"新規{cases['summary']['new']}件、前倒し{cases['summary']['earlier']}件、"
              f"実際に生存{cases['summary']['actually_survived']}件、未判定{cases['summary']['unresolved']}件。", '',
              '関連回帰テストは`logs/e23/tests.log`。既定OFFはE22とNPZ・イベントがバイト一致、診断一致。',
              '入力・モデル・production_config.pyのE22ハッシュ一致は`logs/e23/INPUTS.json`。']
    lines += case_lines(cases['cases'])
    if video:
        mobile = video['mobile'][0]
        lines += ['', '`logs/review_zenchi_g41_43_e23/`（CSV・NPZ・イベント・動画・再現コマンド）。',
            '`D:/puyo_analyzer/videos/review/zenchi_g41-43_e23_mobile.mp4`。',
            f"横1280・CRF28・映像上限1Mbps・音声保持、{mobile['bytes']:,} bytes、{mobile['frames']:,}フレーム。"]
    DOCUMENT.write_text(DOCUMENT.read_text().split(MARKER)[0]+'\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
