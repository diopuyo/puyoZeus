"""再現済み診断結果を全フレーム表と代表時刻表へ整形する。"""
from __future__ import annotations

from collections import Counter
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts._diag_review1003_extract import OUT, ROOT, WINDOWS, resources
from scripts._diag_review1003_inspect import VALUE_COLUMNS

SIDES = ('1P', '2P')
P_COLUMNS = VALUE_COLUMNS[2:]
HEADERS = ['秒', '採用器', 'G_fe', 'S1′', 'S3′暫定', 'S3′', '着弾G_fe', '合成', '採用', '表示']
FINDINGS = '''# レビュー3件の診断（合否外・採否判断なし）

基準コード802efbc、本番R1s+D-v2h2b。時刻は元動画絶対秒、確率は1P視点。
対象はセット1の3試合だけ。全16,632出力フレーム中、421+210+300=931フレーム。
第13・26試合は処理入力と30fps描画の重複保持があるため、931件を独立した観測と扱わない。
空欄は未評価・記録なしであり0ではない。S1′は対象区間全行欠測。
G_feは直近静止評価の保持値。パネルDは最新STABLE盤面、S3モデルDは区間開始時に凍結した別の値。
NON-STABLEの生盤面を直接評価した記録ではない。仮想着弾G_feはR1sではなく既存モデル。

## 直接理由

1. #1：214秒はS3′暫定0.713968と着弾G_fe0.963320の等重みlogit平均=0.890069。
   連鎖得点30950対10540は442対150個、確率層の残受け292個として入っている。
   2P打ち返し余地は-291.571個、NF k1〜k5は0,0,6,6,6。受け量を見ていないわけではない。
   S3送り特徴はn/(1+n)で0.997743対0.993402に圧縮される。木モデルの単一特徴の寄与は断定できない。
   死亡用台帳は154個。死亡証明が成立する更新でもhistory_complete=Falseにより取り消される。
   境界136.133秒でゼロ得点は読めたが、STABLE初期盤面の条件boards_ok=Falseで開始履歴を受理できなかった。
   0.89は固定上限ではない。222秒以降はG_fe約0.99へ移る。

2. #2：947.117秒の切替で確率用受け量が両側0になり、着弾合成を採用せずS3′暫定へ戻る。
   947.217秒は合成0.890677が記録にあるのに採用0.394543、表示0.545618。
   1P完走予測0連鎖/0点・predicted_final_board無し。発火前候補はorigin_differenceで不受理。
   未完走の1Pは累積1880点のみ、2Pは2540点が入り、送り26.857対36.286個として学習モデルへ渡る。
   live countは発火前STABLE盤面のシミュレーションへfallbackし、1P NFは全0。未完走1280点は見えていない。
   さらに凍結Dは938.617秒の交換開始時の文脈で、パネルの現在盤面とは異なる。
   1P得点だけ未来実測3160へ置換する診断実験では0.674140になる（未来情報を使った感度検査であり修正案の評価ではない）。
   死亡用受け量は同時刻21個残る。確率用のイベント差引量と死亡台帳の不一致が切替を生む。

3. #3：1728.983秒はS3′暫定0.852454と着弾G_fe0.943711の合成0.907765。
   2Pの進行中連鎖は4連鎖3460点として得点経路へ入るが、完走後盤面で計算するNFは両側全0。
   これから消える未完走分をNFで再加算しない仕様自体と、その完走予測・時刻が正しいかは別問題。
   1Pは1727.817秒に登録されたchain18が終了信号無しのまま13800点を保有し、確率層は2P受け199個とする。
   同時点の映像・入力では1Pは操作中、予告156個が1P側にある。交換開始以前の相殺・予告を含む整合が崩れている。
   固定Dは1P最大連鎖8/19、2P4/19（最新パネルは1/19、2/19）、発火前k1〜k5は両側1に飽和。
   1P得点を0にする感度検査でもS3は0.697250で、得点だけの修正ではこの凍結文脈の影響が残る。
   1731.750秒にはfinalize通知81348点がchain18へ入り、1732.983秒の予測受け量は2P1415個へ膨らむ。
   実表示差・帰属式累計は数十点であり同じ連鎖の得点として整合しない。上流通知を最新未確定chainへ結ぶ経路が直接原因。
   1734.750秒にactivity_timeoutでG_feへ戻り、1734.983秒は採用0.001734。
   『真の2P勝率』と認識誤りのセル別正解は未ラベル。残り火力の正解量そのものは断定していない。

## 修正候補と副作用（すべて候補、採否なし）

|件|候補・位置|誤確定・表示の飛び・遅延の見込み|
|---|---|---|
|1|src/exchange_history_integrity.py:58,73：境界ゼロと初期盤面の時刻整合・履歴受理を精査|緩和だけなら残像・途中開始を完全履歴と誤認し死亡を誤確定。同期証拠を待つ方式は数フレーム以上の遅延|
|1|src/exchange_event_landing.py:43,426：不足量と証拠品質を入力にした確率合成の再学習|強制0.98化は誤確定相当の表示を増やす。再学習は確率の変動・較正悪化を別途検証する必要|
|2,3|src/exchange_event_landing.py:293,421：確率用の相殺台帳と死亡用台帳の整合|台帳欠測・予測を確定扱いすると誤確定。切替の段差は減る可能性、予測訂正時は逆方向に飛ぶ可能性|
|2|src/exchange_event_overlay.py:578,631：完走0・予測矛盾時の扱いと途中完走回復を明示|不確かな途中盤面を信用すると幻連鎖。保留は低下抑制と引換えに反転・終局の遅延|
|2|src/exchange_event_landing.py:421／src/exchange_display_smoothing.py:95：入力有効性と切替理由を伴う表示遷移|平滑化のみでは誤った0.39を隠す。急落は弱まるが真の逆転も遅れる|
|3|src/exchange_event_tracker.py:378／src/exchange_event_overlay.py:694：finalizeとchainの同一性・得点差整合|誤帰属は減る可能性。正当な遅延通知を拒否すると確定遅延、補正時に表示段差|
|3|src/exchange_event_tracker.py:217／src/exchange_event_overlay.py:631：固定Dとlive countの時間文脈を揃える|モデル学習時と入力分布が変わるため再学習・左右変換検証が必要。細かな盤面更新で表示が揺れる可能性|
|3|src/exchange_event_overlay.py:443／src/exchange_event_death_formula.py:60：途中・完走通知の開始時刻と終了対応を精査|古い通知を新連鎖として再登録すると火力二重算入。結合を強めすぎると本当の新連鎖を見逃し遅延|

## 表の読み方

下表は代表21時点。全931フレームを間引かない表はfull_frames.md、元列全部はgameXX_frames.csv。
gameXX_joined.jsonに各フレーム直前の入力・両盤面・NEXT・会計、replay_gameXX/observations.jsonlに実モデル入力を保存。
盤面文字列は隠し段row0からrow12、各6列。0空/1赤/2青/3緑/4黄/5紫/9おじゃま/?不明。
予告・受け量・余地・NFは個数。最大連鎖はパネルの正規化値×19を表示。NFはk1〜k5順。
'''


def fmt(value: Any) -> str:
    """欠測をゼロにせず、表だけ丸める。"""
    if value is None or value == '':
        return '—'
    if isinstance(value, (tuple, list)):
        return ','.join(fmt(v) for v in value)
    try:
        return f'{float(value):.3f}'.rstrip('0').rstrip('.')
    except (ValueError, TypeError):
        return str(value)


def table(headers: list[str], rows: list[list]) -> str:
    """Markdown全表を出す。"""
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |'] +
        ['| ' + ' | '.join(fmt(v).replace('|', '/') for v in row) + ' |' for row in rows]) + '\n'


def values(rows: list[dict]) -> str:
    """各評価器、合成、採用、表示を同じ行へ並べる。"""
    return table(HEADERS, [[r['t_sec'], r['new_evaluator'], *[r[k] for k in P_COLUMNS]] for r in rows])


def input_table(rows: list[dict]) -> str:
    """側別の物理入力・残り火力を並べる。"""
    output = []
    for r in rows:
        for side in SIDES:
            nf = [r[f'{side}_NF_ojama_k{k}'] for k in range(1, 6)]
            chain = r[f'{side}_d_current_max_chain']
            output.append([r['t_sec'], side, r[f'{side}_state'], r[f'{side}_confirmed_sec'],
                r[f'{side}_chain_count'], r[f'{side}_formula_total'], r[f'{side}_forecast'],
                r[f'{side}_incoming'], float(chain)*19 if chain else None, nf,
                r[f'{side}_counter_margin'], r[f'{side}_hands'],
                r[f'{side}_required_cancel'], r[f'{side}_near_future_send']])
    return table(['秒', '側', '状態', '確定盤面秒', '観測段', '帰属式点', '予告', '確率受け',
                  '最大連鎖', 'NF1〜5', '余地', '応手数', '必要相殺', '応手火力'], output)


def board_table(rows: list[dict]) -> str:
    """両盤面を省略せず、全セルとNEXTを記す。"""
    output = []
    for r in rows:
        for side, board in zip(SIDES, r['inputs']['boards']):
            output.append([r['t_sec'], side, board['t_sec'], board['board'],
                           board['next'], board['dnext']])
    return table(['秒', '側', '盤面秒', '13行×6列', 'NEXT', 'NEXT2'], output)


def transitions(rows: list[dict]) -> list[dict]:
    """採用器の全切替を抽出する。"""
    result, previous = [], None
    for r in rows:
        if r['new_evaluator'] != previous:
            result.append(dict(t_sec=float(r['t_sec']), source=r['new_evaluator'],
                               selected=float(r['p1_selected'])))
            previous = r['new_evaluator']
    return result


def prediction_table(game: int, rows: list[dict]) -> str:
    """後日の確定値を混ぜず、当該時点の完走予測・受け量を表にする。"""
    observed = [json.loads(s) for s in (OUT / f'replay_game{game:02}' /
        'observations.jsonl').read_text().splitlines()]
    stamps = [r['t_sec'] for r in observed]
    events = json.loads((OUT / f'game{game:02}_events.json').read_text())
    landings = sorted([v for e in events for v in e['values'] if 'incoming' in v],
                      key=lambda v: v['t_sec'])
    landing_times = [v['t_sec'] for v in landings]
    output = []
    for row in rows:
        stamp = float(row['t_sec'])
        idx = bisect_right(stamps, stamp) - 1
        current = observed[idx] if idx >= 0 else {}
        li = bisect_right(landing_times, stamp) - 1
        landing = landings[li] if li >= 0 and current.get('chains') else {}
        chains = current.get('chains', [])
        prediction = []
        for side in SIDES:
            chain = next((c for c in reversed(chains) if c['side'] == side), {})
            prediction.append('/'.join(fmt(chain.get(k)) for k in
                ('predicted_chain_count', 'predicted_final_score', 'end_signal_sec')))
        totals = current.get('score_totals', {}).get('array')
        output.append([stamp, *prediction, totals, landing.get('death_incoming'),
            landing.get('completion_certain'), landing.get('resolving_send'),
            landing.get('history_integrity', {}).get('blocked'),
            landing.get('state_safety')])
    return table(['秒', '1P予測段/点/終了秒', '2P予測段/点/終了秒', 'S3入力点1P,2P',
                  '死亡用受け1P,2P', '完走有効', '消去中火力', '履歴で死亡取消', '安全性理由'], output)


def main() -> None:
    """フレーム表・根拠・再現検証を一括保存する。"""
    resource = resources()
    report, full, stats = [FINDINGS], [], []
    for game, start, end in WINDOWS:
        rows = json.loads((OUT / f'game{game:02}_joined.json').read_text())
        samples = json.loads((OUT / f'game{game:02}_samples.json').read_text())
        title = f'\n## 第{game}試合 {start}〜{end}秒（{len(rows)}出力フレーム）\n\n'
        report.extend([title, values(samples), '\n', input_table(samples), '\n',
                       prediction_table(game, samples), '\n', board_table(samples)])
        full.extend([title, values(rows), '\n', input_table(rows), '\n',
                     prediction_table(game, rows), '\n', board_table(rows)])
        expected, actual = ROOT / 'work' / f'set1_game{game:02}', OUT / f'replay_game{game:02}'
        identity = {name: hashlib.sha256((expected/name).read_bytes()).hexdigest() ==
                    hashlib.sha256((actual/name).read_bytes()).hexdigest()
                    for name in ('events.jsonl', 'display.npz')}
        stats.append(dict(game=game, frames=len(rows), transitions=transitions(rows),
            sources=dict(Counter(r['new_evaluator'] for r in rows)), identity=identity))
        assert all(identity.values()), identity
    (OUT / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    (OUT / 'full_frames.md').write_text('\n'.join(full), encoding='utf-8')
    (OUT / 'verification_diag.json').write_text(json.dumps(dict(
        resources=resource, results=stats), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
