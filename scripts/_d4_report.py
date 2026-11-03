"""D4の固定原票から日本語の最終診断書を生成する。"""
from __future__ import annotations
import json
from pathlib import Path
from scripts._d4_loss import OUT, Q

REPORT = Path('docs/D4_R1_EVAL_REGRESSION_20260929.md')


def read(name: str) -> object:
    """D4成果物だけを読む。"""
    return json.loads((OUT/name).read_text())


def top_intervals() -> list[str]:
    """順位は非重複1秒区間の総損失増加で固定する。"""
    result = ['## 1. qの悪化分解', '',
        '固定6,526行、独立勝敗4区間。OFF 0.507977329590 → ON 0.541749642012。',
        '行損失差の総和は220.398111。悪化518行、改善173行、同値5,835行。',
        '下表は非重複1秒窓の上位10。全て独立勝敗区間1、保存game_idx=2、撃ち合い6。各窓の最大悪化行は2P連鎖ID=6（発火218.800秒）を含む。', '',
        '|順位|原動画秒（両端含む）|行数|損失差総和|全6,526行への寄与|直前修正・窓開始までの秒|',
        '|---:|---|---:|---:|---:|---|']
    for i, row in enumerate(read('q_top10_seconds.json'), 1):
        prior = row['preceding']
        result.append(f"|{i}|{row['start']:.3f}–{row['end']:.3f}|{row['n']}|{row['delta_sum']:.6f}|{row['contribution']:.6f}|{prior['signal']} {prior['t']:.3f}、{prior['age']:.3f}秒|")
    result += ['', '9位の218秒窓は開始前の最終修正が215.467秒の1Pおじゃま。区間内218.233秒に主因の2P赤修正が入り、218.800秒の発火まで0.567秒。',
        '5秒窓では220–225秒:+132.684300、225–230秒:+58.606536、215–220秒:+33.151519。',
        'この15秒の合計224.442355が純増の101.83%。他区間の改善が一部を相殺した。', '',
        '|独立勝敗区間 / 保存game_idx|行数|OFF損失|ON損失|全体への寄与|', '|---|---:|---:|---:|---:|']
    for r in read('q_matches.json'):
        result.append(f"|{r['match']} / {r['game']}|{r['n']}|{r['off']:.6f}|{r['on']:.6f}|{r['contribution']:+.6f}|")
    result += ['', '行原票:q_rows.json、連鎖別:q_chains.json、5秒帯:q_intervals.json。',
        '上位区間の盤面・イベント・スナップショット:q_top10_inputs.json、count・隠し段・実モデル特徴:q_7gc4TgFig_trace_examples.json。']
    return result


def mechanism() -> list[str]:
    """主因・反証・介入の範囲を分けて記述する。"""
    baseline = read('q_summary.json')
    trace = read(f'{Q}_trace_summary.json')
    result = ['## 2. 原因と経路別寄与', '',
        '**主因は発火起点の欠落。盤面平均精度の改善と、勝敗を左右する発火点の保持は別だった。**',
        '218.233秒の2Pおじゃま合図で赤(r2,c1)を0→1へ修正。218.300秒にOFFはSTABLEとなって紫(r5,c2)を確定したが、ONはTSUMO_FALLを継続し紫を取得しなかった。',
        '218.333–218.866秒の紫はCNN/HSVとも5（紫）で観測される。218.733秒にONもSTABLEへ戻るが紫は0のまま。',
        '218.800秒の実発火通知は両者同一。直前218.767秒の起点は紫1セルだけ異なり、物理計算はOFF 9連鎖/50,860点、ON 0連鎖/0点。',
        '210.567秒の1P連鎖5も起点が14,010→0点。ただしこちらは同じ採用スナップショットが14,470点へ補う。2P連鎖6は両条件ともfirst_score棄却で補われない。', '',
        '|220.200秒の評価入力・中間値|OFF|ON|', '|---|---|---|',
        '|2P連鎖6 完走予測|50,860点 / 9連鎖|0点 / 0連鎖|',
        '|着弾個数 [1P,2P]|[511,0]|[0,196]|',
        '|2P countの近未来5列（log1p）|全列1.791759（生値5）|全列6.616065（生値746）|',
        '|1P counter margin（符号付きlog1p）|-5.946131|+4.955827|',
        '|隠し段2Pの列0 / 列4|空0.5、4色各0.125|紫1.0 / 赤1.0|',
        '|S3基底p1|0.250203|0.336537|',
        '|着弾後G_fe p1|0.041396|0.900417|',
        '|最終p1|0.107177|0.681690|', '',
        f"全6,526行ではcount差{trace.get('counts',0)}、隠し段分布差{trace.get('hidden',0)}、連鎖属性差{trace.get('chains',0)}、表示由来差{trace.get('source',0)}行。撃ち合いID差{trace.get('exchange',0)}、発火ID/側/時刻差{trace.get('event_identity',0)}行。", '',
        '**合図別介入**: ONの他入力・モデル・評価ロジックを固定し、各発火側の直前修正合図をNEXT/おじゃま/式に分け、該当する物理完走起点だけをOFFへ置換。全6,526行を再採点した。',
        '|OFFへ戻す起点群|変更起点数|q log loss|ONからの損失減少|', '|---|---:|---:|---:|']
    for signal in ('next','ojama','formula'):
        value = read(f'completion_ablation/{signal}/RESULT.json')
        result.append(f"|{signal}|{value['changed_notifications']}|{value['log_loss']:.9f}|{baseline['on']-value['log_loss']:+.9f}|")
    loss = read('completion_ablation/ojama/RESULT.json')['log_loss']
    result += ['', f"おじゃま後2起点の介入で悪化の{(baseline['on']-loss)/(baseline['on']-baseline['off']):.2%}を回復。残差は{loss-baseline['off']:+.9f}。モデル再学習なしで主悪化が消える。", '',
        'これは認識段階の合図そのものを無効化した実験ではない。保存境界には状態機械内部・着地投票履歴がなく、既に変化した後続盤面を真のno-NEXT/no-ojama/no-formulaへ戻せない。従って上記を各合図の総因果効果とは呼ばない。', '',
        '- **H1: 部分支持。** 修正後に認識のSTABLE復帰・取得盤面が変わり、物理完走・count・隠し段へ伝播。主要区間の架空発火増加は認めない。',
        '- **H2: 主因として支持せず、一般的な分布ずれは未確定。** 学習84,445行の列別範囲外を持つqモデル呼出しはOFF 1,293/1,605、ON 1,307/1,605。OFFでも既に多く、ON特有の分布外だけでは主悪化を説明できない。列別範囲検査は多変量の分布検定ではない。',
        '- **H3: 部分支持。** 主因場面は組の片側が先に見える途中でのおじゃま合図修正。その後の着地取得が分岐した。赤の修正値自体は独立後続観測でも一致し、単純な誤色修正ではない。おじゃま189セルは後続一致112・誤修正4・未確認73。NEXTは295/0/3、式は57/0/6。',
        '- **H4: 支持。** qは発火点1セルの欠落→完走0→着弾方向の反転。zenchiは正しくなった起点で相殺予測が採用される一方、死亡反証が未成立のまま残る評価機構。']
    return result


def residual() -> list[str]:
    """母数と時刻対応を省かず、六条件で説明できない残差も残す。"""
    data = read('residual_alignment.json')
    result = ['## 3. 固定379セルが残る理由', '',
        '母数はD3固定3,958置きの同じ保存行・座標・参照色で残る379セル（412→379、33セル減）。各セルを同側・同周期で基準時刻以前の直前R1合図へ対応した。',
        '同時刻の実採否は91/379セル。残り288は過去合図との対応または合図なし。過去合図の理由を「基準時刻で実際に棄却された理由」とは扱わない。時刻差は中央値0.633秒、95%点0.867秒、最大3.133秒。',
        '全体ゲートを優先した排他分類。CNN/HSV、連続一致は上位ゲートを通った場合に判定する。', '',
        '|理由・対応状態|全379セル|同時刻91セル|過去合図または欠測288セル|', '|---|---:|---:|---:|']
    names = ['CNN≠HSV','2フレーム不一致','浮き','上限超','品質不良','連鎖中・非STABLE',
        '周期内消費済み','選択画像が基準と異なる','差分候補外','対応合図なし','修正後に残存・再変化']
    for name in names:
        group = data['groups'].get(name,{})
        a,b = group.get('同時刻',0),group.get('過去合図または欠測',0)
        result.append(f'|{name}|{a+b}|{a}|{b}|')
    result += ['|合計|379|91|288|', '',
        '指定の六条件に対応するものは118/379。残り261/379は、消費済み96、観測時点の違い82、直前合図で差分候補外59、合図なし13、修正後の公開順序/再変化11。',
        '最後の11のうち9セルは、60fps合図の16.7ms前の30fps保存行を固定採点しており、次の保存行では正しい値になる。残る2セルは後続の別の色への再変化。固定母数の379は変更しない。',
        '550修正セルは全合図・全位置の集計であり、同じ早期誤り379セルの修正回数ではない。六つの認識品質ゲートだけを緩めても、主な母数不一致・一回消費・観測時刻差は解決しない。',
        '原票:residual_cells.json、residual_alignment.json、residual_publication.json。']
    return result


def scene() -> list[str]:
    """3分場面のモデル採用と実表示を時系列で記す。"""
    return ['## 4. zenchi 3:00の2P勝率上昇', '',
        '|原動画秒|出来事 / OFF → ON|', '|---:|---|',
        '|2750.833|NEXTで指定4セルを修正。公開は2750.850秒。|',
        '|2751.383|2P連鎖9の発火。スナップショットはOFF起点不一致棄却→ON採用、初回平均予測68,794.536点。|',
        '|2753.317|2Pへの予想着弾1,215.705→253.667個。2P完走候補が新たに相殺へ参加。|',
        '|2755.450|2P予測69,073.962点、1P確率0.759517→0.622752。|',
        '|2758.950–2759.050|2P予測75,440点=1,077個。2Pへの予想着弾1,120→171個、基底p1 0.652837→0.351681、G_fe p1 0.785255→0.848598。|',
        '|2759.050（レビュー3:00）|EMA後の2P実表示27.6063%→36.4443%。|',
        '|2771.250|両条件とも2P≤5%へ初到達。|', '',
        'ONは正しくなった起点から2Pの反撃を見込んでいる。その相殺効果が基底勝率を大きく引き上げ、着弾後G_fe側の逆向き効果を上回る。',
        '3:00時点の死亡判定は両条件ともdead_sides=[]。ONの複数着弾判定はunverified_attack、state_safetyはcompletion_prefix_unverifiedで、負け確定へ振り切れない。隠し段候補の採用・相殺と死亡側の証明条件が別になっているため。',
        'この場面の上昇を「盤面修正が偽の発火を生んだ」とは判定しない。原票:scene_inputs.json、review_trace_examples.json。']


def main() -> None:
    """指定文書以外には書き込まない。"""
    content = ['# D4 R1本番評価悪化の診断（2026-09-29）', '',
        '対象:58ae8a7。診断専用。src/・本番設定・既存入力を変更していない。重い処理はWSL nice 19、BLAS/OMP各1スレッド。CX1/CX2には介入しない。',
        '結論:q悪化の98.54%は、おじゃま合図後に変わった発火起点を物理完走計算へ渡す経路で説明できる。zenchiは起点修正で反撃の相殺が採用される一方、死亡側の確定に至らない。', '']
    for part in (top_intervals(), mechanism(), residual(), scene()):
        content.extend(part+[''])
    content += ['## 再現と成果物', '',
        '全JSON参照はlogs/d4/基準。q/reviewの観測付きOFF/ON再生4件は、元のR1とdisplay.npz・events.jsonlがバイト一致、diagnosticsも一致。',
        'scripts._d4_launch → _d4_loss → _d4_events / _d4_residual / _d4_origins / _d4_compare / _d4_timeline → _d4_findings → _d4_ablation_launch → _d4_report → _d4_validate。',
        '各実行は worktreeをcwdとして `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 <既存venv>/bin/python -B -m scripts._d4_...`。',
        '学習範囲原票:training_support.json。合図ごとの真の認識無効化には再認識が必要なため、この診断の起点介入値と混同しない。']
    REPORT.write_text('\n'.join(content)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
