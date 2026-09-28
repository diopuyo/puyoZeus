"""E31の段0、取得率、事前登録合否を原票から報告する。"""
from __future__ import annotations
import json
from pathlib import Path
from scripts.run_e31 import report, OUT, SOURCES
from scripts.report_e30 import format_quantile
from scripts.run_e3_exchange_eval_20260926 import save_json

REPORT = Path('docs/E31_PREFIRE_SNAPSHOT_20260928.md')


def scene_notes() -> list[str]:
    """目視画像とセル別差分を示し、不可視段を正解扱いしない。"""
    return ['## 段0: 最後の1手だけの不足ではない', '',
        '起点2751.383333秒、2P連鎖ID9。2750.183333〜2751.383333秒を60fpsで73枚保存。',
        '`logs/e31/scene/frames.json` は全フレームのCNN融合・CNN単独・HSV単独の盤面と保存起点との差分。',
        '各PNGは原動画の2P盤面を1920×1080座標で切り出した実画面。',
        '2750.566667秒の着地前画像で、保存起点には既に可視4セルの不一致がある。',
        '座標はrow/colとも0始まり。(1,3) 空→赤、(1,4) 赤→空、(1,5) 空→緑、(2,5) 空→赤。',
        'すなわち赤1個の列ずれと、右端上部の赤・緑の欠落。盤面下部は一致する。', '',
        '![最後の組ぷよが入る前](../logs/e31/scene/165034_2750.566667.png)', '',
        '2751.000000〜2751.350000秒では、さらに(1,2),(2,2)へ最後の黄黄1組が入る。',
        'この発火前画像のCNN単独・CNN融合・HSVは一致。可視差分6セル、隠し段UNKNOWN4セル。', '',
        '![最後の組ぷよが入った発火前盤面](../logs/e31/scene/165060_2751.000000.png)', '',
        '観測盤面を参考シミュレーションすると13連鎖75180点（実75440点との差260点）。',
        '初段100・2段目320点は一致するが、3段目640点は実900点と一致しない。',
        'UNKNOWNを含む参考値は予測へ採用しない。不可視段の真値は画像から確定できない。',
        '入口の誤りを確認したが、UNKNOWN禁止と差分4セル上限を緩めて対象場面を通すことはしない。', '']


def gates(value: dict) -> list[str]:
    """E27と同じ母数を明示する四条件。"""
    base = json.loads(Path('logs/e27/on/METRICS.json').read_text())
    q,z,d,s = (value[k] for k in ('q','zenchi','deaths','scene'))
    rows = [('q','q log loss',f"{base['q']['log_loss']:.9f}",f"{q['log_loss']:.9f}",f"{q['frames']}行/{q['matches']}試合",'≤.512778'),
        ('zenchi','zenchi一致率',f"{base['zenchi']['agreement']*100:.4f}%",f"{z['agreement']*100:.4f}%",f"{z['hits']}/{z['frames']}",'≥84.68%'),
        ('deaths','誤発火',f"{base['deaths']['false']}/{base['deaths']['total']}",f"{d['false']}/{d['total']}",f"未判定{d['unlabelled']}",'≤1/28'),
        ('scene','2P≤5%初到達',str(base['scene']['first_sec']),str(s['first_sec']),'1場面','≤2766.0秒')]
    return ['|指標|E27|E31|母数|基準|判定|','|---|---|---|---|---|---|']+[
        f"|{label}|{before}|{after}|{n}|{limit}|{'合格' if value['gates'][key] else '不合格'}|"
        for key,label,before,after,n,limit in rows]


def acquisition(counts: dict) -> list[str]:
    """最初の棄却理由で分類し、撤回も採用時誤差の分母に含める。"""
    lines = ['## 取得率と最終得点のずれ', '', '|動画|全発火|初回採用|保持|撤回|不採用理由|誤差P50/P95|完全一致|',
             '|---|---:|---:|---:|---:|---|---|---|']
    labels = dict(no_landed_window='窓不足', unknown='UNKNOWN', origin_difference='起点差分',
        missing_first_formula='初段式未取得', first_score='初段得点不一致')
    for source in (*SOURCES,'three_videos'):
        row = counts['three_videos'] if source == 'three_videos' else counts['sources'][source]
        reasons = '、'.join(f'{labels.get(k,k)}{v}' for k,v in row['rejected'].items())
        lines.append(f"|{source}|{row['fires']}|{row['accepted']}|{row['retained']}|{row['withdrawn']}|"
            f"{reasons}|{format_quantile(row['score_absolute_error'])}|{row['exact']}/{row['score_absolute_error']['n']}|")
    return lines+['','全発火はE27の全連鎖ID（0連鎖起点だけへの限定なし）。レビューとzenchiは合計へ重複加算しない。',
        '不採用は窓不足→UNKNOWN→試合色→浮遊→起点差分→初段式の順で最初の理由を記録。',
        '得点差は採用時予測と実最終得点の絶対差。途中撤回も含める。',
        f"採用後の実最終得点が未確定: {counts['three_videos']['unresolved']}件（誤差の母数から除外）。",'']


def main() -> None:
    """原票を集約し、合格時だけ動画作成可能とする。"""
    value = report()
    counts = json.loads((OUT/'SNAPSHOT_COUNTS.json').read_text())
    lines = ['# E31: 発火前画像の予測専用スナップショット', '',
        '判定: '+('合格' if value['candidate'] else '不合格、採用候補E27を維持'), '',
        'E27 87b0fca条件＋既定OFF `--prefire-snapshot`。production_config.py変更なし。', '',
        *scene_notes(), *gates(value), '', *acquisition(counts),
        '## 実装と再現', '',
        '取得窓1.2秒・30Hz、着地形状3フレーム以上でセル多数決。同数はUNKNOWN。',
        'AnimationFilterとeffect_glow_detectorで点滅・煙・バーストの品質不良を除外。',
        '浮遊除外後の最大占有数の最後の着地区間を使用し、消去後の盤面を避ける。',
        '差分修正上限MAX_CORRECTION_CELLS=4。試合色はSTABLE盤面から因果的に取得。',
        '初段一致で単一連鎖を採用し、途中の得点不一致または観測段欠落で永久撤回。',
        'E27更新前に前回の予測を戻すため、現在層・確定履歴・STABLE判定は不変。',
        '完走得点・送り量・盤面はS3/着弾/死亡予測へ渡し、画面/CSV/原票に「発火前盤面から予測」を付与。',
        '学習特徴の新規列は追加しない。既存のS3差分反転・正規化・count変換経路を維持。',
        '`bash scripts/_launch_e31_enrich.sh` → `bash scripts/_launch_e31.sh` →',
        '`python -m scripts.report_e31` → `python -m scripts.verify_e31`。',
        'OFFは追加観測付き入力で全5記録を再生し、E27のNPZ全列/全バイト・イベント原票・診断JSONと照合。',
        'ONで一度採用した後の表示EMAは従来の履歴を引き継ぐ。未採用の生確率と表示平滑化の履歴差は分けて検証。',
        '関連回帰667成功・1スキップ、最終差分のライブ取得/CSV/記録互換118成功。',
        '原票: logs/e31/tests.log、final_tests.log、VERIFICATION.json。', '',
        '## 動画', '', '全条件合格時だけ指定のE31レビュー動画を生成。'+
        ('合格。' if value['candidate'] else '今回は不合格のため生成なし。')]
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    save_json(OUT/'SUMMARY.json',dict(metrics=value,counts=counts,report=str(REPORT)))


if __name__ == '__main__':
    main()
