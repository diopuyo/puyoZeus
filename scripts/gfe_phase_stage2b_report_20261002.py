"""段2bの全採点表・用途別時間・経路別寄与を統括へ渡す。"""
from __future__ import annotations

import json
from scripts.gfe_phase_stage2b_train_20261002 import OUT
from scripts.gfe_phase_stage2b_score_20261002 import MODES
from scripts.gfe_phase_stage2_report_20261002 import pair


def comparisons(result: dict) -> list[str]:
    """合算と各セットの両指標を省略せず出力する。"""
    lines = ['|構成|区分|指標|本番|候補|差|95%区間|試合数|',
             '|---|---|---|---:|---:|---:|---|---:|']
    for mode in MODES[1:]:
        for metric, key in [('M_time', 'overall'), ('M_agree', 'overall_agree')]:
            lines.append(f"|{mode}|合算|{metric}|{pair(result['decisions'][mode][key])}|")
        for name in ('set1', 'set2'):
            for metric in ('m_time', 'agree'):
                lines.append(f"|{mode}|{name}|{metric}|{pair(result[name]['comparisons'][mode][metric])}|")
    return lines


def regression(result: dict) -> list[str]:
    """母数・反転・誤確定・3:00場面・第14試合を示す。"""
    lines = ['', '|区分|試合数|全表示行|M_time評価行|終盤評価行|', '|---|---:|---:|---:|---:|']
    for name in ('set1', 'set2'):
        d = result[name]
        lines.append(f"|{name}|{d['games']}|{d['display_frames']['baseline']}|"
                     f"{d['evaluated_frames']['baseline']}|{d['late_frames']['baseline']}|")
    lines += ['', '|区分|指標|本番|V3s|R1s|', '|---|---|---:|---:|---:|']
    for name in ('set1', 'set2'):
        for metric in ('flips', 'false_deaths'):
            values = '|'.join(str(result[name][metric][m]) for m in MODES)
            lines.append(f'|{name}|{metric}|{values}|')
    for key in ('cases', 'false', 'scene_first_sec', 'game14'):
        values = '|'.join(str(result['regression'][m][key]) for m in MODES)
        lines.append(f'|既存5記録|{key}|{values}|')
    return lines


def latency(result: dict) -> list[str]:
    """前回同様の通知合計と用途別の通知合計を区別する。"""
    lines = ['', '|構成|用途|通知数|呼出数|G_fe P95(ms)|推論P95(ms)|最大(ms)|',
             '|---|---|---:|---:|---:|---:|---:|']
    for mode, d in result['latency'].items():
        rows = [('全用途', dict(d, calls=d['gfe_calls']))] + list(d.get('by_purpose', {}).items())
        for purpose, v in rows:
            lines.append(f"|{mode}|{purpose}|{v['notifications']}|{v['calls']}|{v['p95_ms']:.6f}|"
                         f"{v['inference_p95_ms']:.6f}|{v['maximum_ms']:.6f}|")
    lines += ['', '時間は段2と同じ定義（G_fe特徴変換＋推論、M0/D生成は含めない）。'
              '全用途はG_feが発生した通知、用途別はその用途が発生した通知が母数。'
              '同一通知の複数呼出しを合算。本番の時間は段2の保存値を再利用。']
    return lines


def routes(result: dict) -> list[str]:
    """診断と同じ重み・足し戻せる寄与・試合bootstrap区間を並べる。"""
    lines = ['', '|構成|経路|表示行|重み質量|本番LL|候補LL|経路内差|全体寄与|寄与95%区間|平均確率差|',
             '|---|---|---:|---:|---:|---:|---:|---:|---|---:|']
    notes = []
    for mode, values in result['routes'].items():
        for v in values['rows']:
            lines.append(f"|{mode}|{v['route']}|{v['frames']}|{v['mass']:.6f}|{v['baseline']:.9f}|"
                f"{v['candidate']:.9f}|{v['difference']:+.9f}|{v['contribution']:+.9f}|"
                f"[{v['lower']:+.9f}, {v['upper']:+.9f}]|{v['probability_delta']:+.9f}|")
        notes += [f"{mode}: 経路変更 {values['route_changes']}行。寄与合計={values['difference']:+.9f}。", '']
    return lines + [''] + notes


def gates(result: dict) -> list[str]:
    """旧門と門v2を混ぜず、機械判定だけを報告する。"""
    lines = ['', '|構成|条件1|条件2′|条件3′|条件4|条件6|門v2全条件|旧条件2|旧条件3|旧全条件|',
             '|---|---|---|---|---|---|---|---|---|---|']
    for mode, d in result['decisions'].items():
        keys = ('condition1', 'condition2_prime', 'condition3_prime', 'condition4', 'condition6')
        values = [d['gates_v2'][k] for k in keys] + [d['passed_v2'], d['gates']['condition2'],
                    d['gates']['condition3'], d['passed']]
        lines.append(f'|{mode}|'+'|'.join('成立' if v else '不成立' for v in values)+'|')
    return lines


def write_summary(result: dict) -> None:
    """全表を保存し、採用とは区別した結果を明示する。"""
    lines = ['# G_fe 段2b（2026-10-02）', '', '採否は統括。本番登録なし。', '',
        '事前登録: PREREGISTRATION.md。V3s=静止54列HistGBC、R1s=静止59列HistGBC。'
        '両候補の仮想着弾は現行LR。M0・S1′・S3′・表示平滑・証明器・再生構成は段2と同じ。',
        '本番18記録は段2の保存値を再利用。115試合（57+58）の試合単位対bootstrap 10,000回、seed 0。',
        '同じ評価セットの診断を見た後の構造変更であり、選択の楽観が残りうる。'
        '合格しても独立再計算・差分査読と統括の判断が必要。', '']
    lines += comparisons(result) + regression(result) + latency(result) + routes(result) + gates(result)
    lines += ['', '|既定OFF照合記録|display.npz|events.jsonl|', '|---|---|---|']
    for source, values in result['verification']['byte_identity'].items():
        lines.append(f"|{source}|{'一致' if values['display.npz'] else '不一致'}|"
                     f"{'一致' if values['events.jsonl'] else '不一致'}|")
    train = json.loads((OUT/'TRAIN_R1.json').read_text())
    lines += ['', f"R1学習: {train['rows']:,}行 / {train['games']:,}試合 / {train['videos']}動画 / "
              f"{train['seconds']:.2f}秒。保存先 models/exchange_event_v1_r1_hgb/。", '',
              '各再生の gfe_usage.npz は用途・モデル・通知・時刻・計算時間の呼出し原票、'
              'GFE_USAGE.json はモデルSHA-256と用途別呼出数。経路別・試合別・経路遷移の全表は同名CSV。']
    if (OUT/'TIMING.json').exists():
        timing = json.loads((OUT/'TIMING.json').read_text())
        lines += ['', f"再生・採点の経過時間: {timing['seconds']:.2f}秒（最大{timing['workers']}並列）。"]
    if (OUT/'VERIFICATION.json').exists():
        verification = json.loads((OUT/'VERIFICATION.json').read_text())
        resources = verification['resources']
        lines += ['', f"独立検算: {'一致' if verification['passed'] else '不一致'}。"
            f"資源監視{resources['samples']}回、最小空き{resources['minimum_available_gib']:.3f}GiB、"
            'nice 19・BLAS1。既存モデル・本番設定・固定入力のハッシュ不変。']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
