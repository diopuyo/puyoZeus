"""E22の事前登録判定・保留内訳・レビュー検証を文書へ反映する。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e22')
DOCUMENT = Path('docs/E22_DEATH_FORMULA_2026-09-28.md')
MARKER = '## 最終結果'
SCENE = (2633.8, 2634.75)


def baseline_scene() -> dict:
    """旧候補にも同じ29行の窓を適用する。"""
    data = np.load('logs/e19/hold/review/display.npz')
    mask = (data['t_sec'] >= SCENE[0]) & (data['t_sec'] <= SCENE[1])
    return dict(frames=int(mask.sum()), minimum=float(data['display_p1'][mask].min()))


def report() -> None:
    """実発火母数と未確定を省かず、丸めで合否を変えない。"""
    baseline = json.loads(Path('logs/e19/hold/METRICS.json').read_text())
    baseline['scene'] = baseline_scene()
    result = json.loads((OUT/'on/METRICS.json').read_text())
    holds = json.loads((OUT/'HOLDS.json').read_text())
    complete = Path('logs/review_zenchi_g41_43_e22/complete.json')
    video = json.loads(complete.read_text()) if complete.exists() else None
    save_json(OUT/'METRICS.json', dict(baseline=baseline, result=result, holds=holds['three_videos'], video=video))
    lines = [MARKER, '', '|指標・母数|E19|E22|判定|', '|---|---:|---:|---|']
    for label, key, metric, denominator in [('q log loss', 'q', 'log_loss', '6,526行・4試合'),
            ('zenchi一致率', 'zenchi', 'agreement', '8,333行'), ('指定場面の最小勝率', 'scene', 'minimum', '29行')]:
        gate = 'scene' if key == 'scene' else key
        lines.append(f"|{label}（{denominator}）|{baseline[key][metric]:.8f}|{result[key][metric]:.8f}|{'合格' if result['gates'][gate] else '不合格'}|")
    a, b = baseline['deaths'], result['deaths']
    lines += [f"|誤発火／実発火|{a['false']}/{a['total']}|{b['false']}/{b['total']}|{'合格' if result['gates']['deaths'] else '不合格'}|", '',
              '**固定評価合格**。' if result['candidate'] else '**不合格**。E19候補を維持する。', '',
              '|記録集合|保留|観測で本物と確定|破棄|未確定|', '|---|---:|---:|---:|---:|']
    values = [('3動画合計', holds['three_videos'])]+[(s, v['summary']) for s, v in holds['sources'].items()]
    for source, row in values:
        lines.append(f"|{source}|{row['held']}|{row['confirmed']}|{row['discarded']}|{row['unresolved']}|")
    lines += ['', 'zenchiとreviewは重複があるため3動画合計に加算しない。確認種別・破棄理由・全時刻は`HOLDS.json`。',
              '既定OFFはE19の6,848更新・5,948表示行でNPZとイベントがバイト一致、診断も一致。',
              '回帰テストの件数は`logs/e22/tests.log`。本番設定・モデル・固定入力のハッシュは`INPUTS.json`。']
    if video:
        mobile = video['mobile'][0]
        lines += ['', '`logs/review_zenchi_g41_43_e22/`（CSV・NPZ・イベント・動画・再現コマンド）。',
                  '`D:/puyo_analyzer/videos/review/zenchi_g41-43_e22_mobile.mp4`。',
                  f"横1280、CRF28、映像上限1Mbps、音声保持、{mobile['bytes']:,} bytes、{mobile['frames']:,}フレーム。"]
    else:
        lines += ['', '動画は固定評価合格時のみ生成する。現時点の完了記録はない。']
    DOCUMENT.write_text(DOCUMENT.read_text().split(MARKER)[0]+'\n'+'\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    report()
