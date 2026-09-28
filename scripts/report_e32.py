"""E32の取得率・自己較正・固定合否を保存原票から報告する。"""
from __future__ import annotations
import json
from pathlib import Path
from scripts.run_e32 import OUT, e31, report
from scripts.run_e3_exchange_eval_20260926 import SOURCES, save_json

REPORT = Path('docs/E32_HIDDEN_ROW_BELIEF_20260928.md')


def read(path: Path) -> dict:
    """UTF-8の集計原票を読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def gate_rows(value: dict) -> list[str]:
    """要求された母数付き合否表をE27・E31と並べる。"""
    comparison = read(OUT/'COMPARISON.json')
    q,z,d,s = (value[k] for k in ('q','zenchi','deaths','scene'))
    records = [('q', 'q log loss', lambda v:f"{v['q']['log_loss']:.9f}", f"{q['frames']}行/{q['matches']}試合", '≤.512778'),
        ('zenchi','zenchi一致率',lambda v:f"{v['zenchi']['agreement']*100:.4f}%",f"{z['hits']}/{z['frames']}",'≥84.68%'),
        ('deaths','誤発火',lambda v:f"{v['deaths']['false']}/{v['deaths']['total']}",f"未判定{d['unlabelled']}",'≤1/28'),
        ('scene','2P≤5%初到達',lambda v:str(v['scene']['first_sec']),f"{s['scenes']}場面",'≤2766.0秒')]
    return ['|指標|E27|E31|E32|母数|基準|合否|','|---|---|---|---|---|---|---|']+[
        '|'+ '|'.join([label, *(formatter(comparison[e]) for e in ('e27','e31','e32')), n, limit,
            '合格' if value['gates'][key] else '不合格'])+'|' for key,label,formatter,n,limit in records]


def calibration_lines(cal: dict) -> list[str]:
    """正解ラベルを作らず、後続観測で検証できたセルだけを示す。"""
    lines = ['## 隠し段の自己較正', '', '|動画|的中/母数|当たり率|log loss|', '|---|---|---|---|']
    for name in (*SOURCES,'three_videos'):
        row = cal['three_videos'] if name == 'three_videos' else cal['sources'][name]
        accuracy = f"{row['accuracy']*100:.2f}%" if row['n'] else '対象なし'
        loss = f"{row['log_loss']:.6f}" if row['n'] else '対象なし'
        lines.append(f"|{name}|{row['hits']}/{row['n']}|{accuracy}|{loss}|")
    lines += ['', '|最尤確信度区間|母数|平均確信度|的中率|', '|---|---|---|---|']
    for row in cal['three_videos']['bins']:
        if row['n']:
            lines.append(f"|{row['lower']:.1f}–{row['upper']:.1f}|{row['n']}|{row['confidence']:.4f}|{row['accuracy']:.4f}|")
    return lines+['', '一段消去後の可視盤面が全セル一致する落下だけを対象とし、更新前分布を採点。',
        '中間画像は2回一致が必要。最尤同率は空・色コード順で固定。log lossの下限確率は1e-12。', '']


def acquisition_lines(counts: dict) -> list[str]:
    """E31からのUNKNOWN減少と別の棄却理由への移行を隠さず残す。"""
    old = read(Path('logs/e31/SNAPSHOT_COUNTS.json'))
    lines = ['## 取得率と採用時の得点差', '', '|動画|E31採用|E32採用/全発火|保持/撤回|UNKNOWN E31→E32|不採用理由|誤差P50/P95|完全一致|',
             '|---|---|---|---|---|---|---|---|']
    for name in (*SOURCES,'three_videos'):
        a = old['three_videos'] if name=='three_videos' else old['sources'][name]
        b = counts['three_videos'] if name=='three_videos' else counts['sources'][name]
        errors = b['score_absolute_error']
        quantiles = f"{errors['p50']:.3f}/{errors['p95']:.3f}" if errors['n'] else '対象なし'
        lines.append(f"|{name}|{a['accepted']}|{b['accepted']}/{b['fires']}|{b['retained']}/{b['withdrawn']}|"
            f"{a['rejected'].get('unknown',0)}→{b['rejected'].get('unknown',0)}|{b['rejected']}|"
            f"{quantiles} (n={errors['n']})|{b['exact']}/{errors['n']}|")
    return lines+['', '採用時の加重平均得点と後日確定した実最終得点の絶対差。撤回例も誤差に含む。',
        f"実最終得点未確定は{counts['three_videos']['unresolved']}件。review/zenchiは3動画母数に重複加算しない。", '']


def scene_lines() -> list[str]:
    """指定された連鎖ID9の採否を、全体の取得率と区別して示す。"""
    rows = json.loads((OUT/'SCENE.json').read_text(encoding='utf-8'))
    return ['## 対象場面', ''] + [
        f"2P連鎖ID{r['chain_id']}、起点{r['trigger_sec']:.6f}秒: 採用={r['accepted']}、理由={r['reason']}。"
        for r in rows] + ['', 'E31で確認された可視6セルの差分は、隠し段補完後も可視差分上限4セルを超える。', '']


def main() -> None:
    """固定閾値に基づく結果と再現手順を文書化する。"""
    value = report()
    cal, counts = read(OUT/'CALIBRATION.json'), read(OUT/'SNAPSHOT_COUNTS.json')
    timings = {s:read(e31.prior.directory('on',s)/'TIMING.json') for s in SOURCES}
    lines = ['# E32: 履歴からの隠し段確率盤面', '', '判定: '+('合格' if value['candidate'] else '不合格、E27を維持'), '',
        *gate_rows(value), '', *scene_lines(), *calibration_lines(cal), *acquisition_lines(counts),
        '## 所要時間', '', '|動画|更新P50 ms|更新P95 ms|母数|更新合計秒|', '|---|---|---|---|---|']
    for name,row in timings.items():
        times = row['update_sec']
        lines.append(f"|{name}|{times['p50']*1000:.3f}|{times['p95']*1000:.3f}|{times['n']}|{row['total_sec']:.3f}|")
    enumeration = read(OUT/'ENUMERATION_TIMING.json')
    lines += ['', f"列挙P50/P95: {enumeration['p50']*1000:.3f}/{enumeration['p95']*1000:.3f} ms（n={enumeration['n']}）。", '', '## 実装・再現', '',
        'ProbabilisticCellを再利用。既存hidden_row_probabilityはおじゃまを含む静的事前のみのため変更しない。',
        '新外部wrapperはSTABLE確定盤面・読んだNEXT・2回一致の途中画像を時系列処理する。',
        '試合4色＋空、天井未到達は空。情報なしの満杯列は占有事前0.5と4色一様。',
        '読んだNEXTの配置と可視増分を照合し、隠し段へ入った色・列を周辺化する。',
        '発火前画像の終了時刻以前の履歴を複製し、隠し段だけを累積99%・上限4096で列挙。',
        'E31の可視UNKNOWN棄却・可視差分4セル制限を維持。実測式と不一致の候補は段ごとに永久除外。',
        'S3/着弾は候補別モデル勝率の平均。終端は重み過半時のみ。死亡は最大火力・全終端で楽観的に反証。',
        '新しい学習特徴列なし。1P/2P順の候補得点を既存の対称化・正規化・count経路へ渡す。',
        '全候補消滅時はE27の同フレーム予測へ撤回。ONの表示EMAはそれまでの表示履歴を保持する。',
        '本番設定変更なし。OFFはE27条件で全5記録の全列・イベント原票・診断を照合。',
        '`bash scripts/_launch_e32.sh` → `python -m scripts.verify_e32` → `python -m scripts.report_e32`。', '',
        '対象場面原票: `logs/e32/SCENE.json`。較正原票は各`snapshot_final_audit.json`。',
        '検証: `logs/e32/VERIFICATION.json`、`logs/e32/tests.log`。', '', '## 動画', '',
        '合格時のみ指定先へ生成。'+('合格。' if value['candidate'] else '今回は不合格のため生成なし。')]
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save_json(OUT/'SUMMARY.json', dict(metrics=value, calibration=cal, counts=counts, timings=timings, report=str(REPORT)))


if __name__ == '__main__':
    main()
