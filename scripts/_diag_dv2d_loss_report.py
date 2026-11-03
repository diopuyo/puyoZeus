"""診断の全表・個別入力・修正案を生成する。採否・合否は記載しない。"""
from __future__ import annotations

from collections import Counter
import gzip
import json
from typing import Any

from scripts._diag_dv2d_loss_inventory import OUT, read, save, resource
from scripts._diag_dv2d_loss_inputs import recognition_delta

CODES = dict(color_score_mismatch='M', board_unsettled='B', completion_prefix_unverified='C',
    invalid_bound_input='I', optimistic_survivor='S', bound_limit='L',
    unverified_attack_or_budget='U', missing_post_counter_completion='P',
    missing_majority_board='F', unknown_board='X', missing_board_score='Z',
    all_responses_dead='死亡証明', post_counter_upper_bound='死亡証明')
FIXES = [
 ['D1','src/exchange_single_death_proof.py:27 / src/exchange_landing_safety.py:31',
  '拒否理由を分ける。board_unsettledはNEXTだけの変化と盤面変化を別時計にし、color_score_mismatchは独立した得点・消去観測で比較基準を再同期する。旧死亡候補を無条件保持する修正にはしない。',
  '高：不整合入力のlegacy_kept復活は元の誤確定経路を直接戻す。静穏待ち短縮も操作ぷよや残像を固定盤面と誤認し得る。'],
 ['E35″','src/exchange_post_counter_bound.py:102 / src/post_counter_death_bound.py:78',
  '手持ちが実測できた場合だけ未知ツモを既知集合へ絞る。CHAIN中は発火設置・NEXT不変・終了未観測の証拠を再検討し、未知手持ち一手自体は一律に削除しない。',
  '高：未知ツモ省略は生存できる打ち返し枝を落とす。上界の緩さを減らすなら全消し持越し・色供給・手数の上限証明が必要。'],
 ['H1','src/exchange_history_integrity.py:23 / scripts/visualize_advantage_overlay.py:7552',
  '連続記録で受理された得点リセットと、過去の観測死亡・得点ゼロ3フレームを開始証拠に追加する。記録の先頭では境界以前の台帳を持ち越すか、その境界から再生する。',
  '中〜高：OCRの誤リセットを開始にすると欠落台帳を信用する。confirmed_dead_sidesには検出理由がなく、独立した勝利数パネルの裏付けと同一視できない。'],
 ['D3′','src/exchange_death_integrity_v2.py:13 / src/exchange_event_tracker.py:437 / src/exchange_pending_ledger.py:69',
  '式・確定通知のchain_id帰属とscore_before取得を改善する。独立根拠で裏付けられる攻撃下限と超過分を分け、古い連鎖の小さな得点差で残予告全体が未検証になる経路を調べる。相殺履歴を含む下限再計算が必要。',
  '高：前連鎖の式・得点や次連鎖の通知を再利用すると過大攻撃量を証明し、別の誤確定を生む。'],
 ['D4','src/board_state_machine.py:2025 / src/placement_inferrer.py:983',
  '追加候補ごとにslide_motion・placement_validatedと拒否セルを記録し、正しい設置終了の見落としを補う。空セルの再占有には設置イベントまたは物理遷移の証拠を要求する。',
  '高：色一致だけの救済は操作中ぷよ・通過ぷよ・残像の再固定を戻す。今回の一つ抜きは認識記録差し替えなので、特定の拒否セルの因果は別途認識側のログが必要。'],
 ['その他','src/post_counter_death_bound.py:149 / src/exchange_event_landing.py:306',
  'invalid_bound_inputをUNKNOWN・浮遊・既死・色集合・手数の各理由に分割し、bound_limitはキャッシュや証明探索の改善で減らす。観測死亡は連鎖中の盤面傾斜・終了演出の検出時刻を別記録する。',
  '中〜高：入力不正や探索未完了を死亡へ読み替える変更は危険。検出遅延の短縮も連鎖演出と死亡演出の誤識別に注意が必要。']]


def table(columns: list[str], rows: list[list[Any]]) -> str:
    """表セルの改行と区切りをエスケープする。"""
    def cell(value: Any) -> str:
        if value is None:
            return '—'
        if isinstance(value, float):
            return f'{value:.3f}'
        return str(value).replace('|', '／').replace('\n', ' ')
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    lines += ['| '+' | '.join(cell(v) for v in row)+' |' for row in rows]
    return '\n'.join(lines)+'\n\n'


def label(row: dict) -> str:
    """集合名と人間向け試合番号を組み合わせる。"""
    group = dict(set1='セット1', set2='セット2', fcXG83vInDY='fc', q_7gc4TgFig='q')[row['group']]
    return f"{group} 第{row['game']}試合"


def condition(why: dict) -> str:
    """一つの評価時点の実拒否を、旗の有効化だけとは分けて表す。"""
    parts = []
    if why.get('h1'):
        parts.append('H1')
    if why.get('single'):
        parts.append('D1:'+CODES.get(why['single'], why['single']))
    post = why.get('post_proofs') or ([why['post']] if why.get('post') else [])
    if post:
        parts.append(('E35″:' if why.get('unknown') else '証明:')+','.join(CODES.get(p, p) for p in post))
    return '＋'.join(parts) or '評価なし'


def evidence(row: dict) -> str:
    """原票の入力と回復の実測を短く併記する。"""
    first = row['first_reasons']
    restored = ', '.join(f"{flag}@{value['first_sec']:.3f}"
        for flag,value in row['recovery'].items() if value['first_sec'] is not None)
    seen = list(dict.fromkeys(condition(run['signature']) for run in row['reason_series']))
    later = '→'.join(seen[:4]) + ('→…' if len(seen)>4 else '')
    unknown = '/'.join(first.get('unknown_causes', []))
    detail = f'；未知手持ち条件 {unknown}' if unknown else ''
    return (f"{row['record']}/idx{row['local_game']}；受{first.get('incoming')}/手{first.get('hands')}；"
            f"系列 {later}；単独解除 {restored or 'なし'}{detail}")


def cases_table(rows: list[dict]) -> str:
    """55件を省略せず列挙する。"""
    values = []
    for row in rows:
        name = f"{row['group']}_{row['game']}"
        values.append([f'[{label(row)}](cases/{name}.json)', row['side'], row['first_sec'],
            row['candidate_observed_death_sec'], row['delay_sec'], condition(row['first_reasons']), evidence(row)])
    return table(['試合','側','本番確定秒','v2d観測死亡秒','遅れ秒','本番時刻の拒否条件','入力・後続系列・解除再生の初回秒'], values)


def aggregate_tables(data: dict, rows: list[dict]) -> str:
    """重複する解除効果と、排他的な分類を別々に表示する。"""
    aggregate = data['aggregate']
    values = []
    for flag, result in aggregate['effects'].items():
        selected = [r for r in rows if r['recovery'][flag]['first_sec'] is not None]
        risk = data['risks'][flag]
        values.append([flag, result['count'], sum(r['delay_sec'] for r in selected), result['restored_sec'],
                       sum(v['false_events'] for v in risk.values()), sum(v['false_frames'] for v in risk.values())])
    text = '一つ抜きの回復効果（列ごとの対象は重複）。秒数は初回確定の先行時間であり、表示継続時間ではない。\n\n'
    text += table(['外した旗','回復件数','対象の元遅れ合計秒','戻った秒','勝者側への計算確定件数','誤確定表示フレーム'], values)
    text += '最後の2列は今回再生した接頭区間だけの参考監査。件数は表示外を含む。既存全体評価の再採点ではない。\n\n'
    text += '相互排他的分類。「または」はどちらか一旗を外した再生で回復した意味。\n\n'
    text += table(['単独解除で回復した旗の組','件数','元遅れ合計秒'],
        [[key.replace('+',' または '), value['count'], value['total_delay_sec']]
         for key,value in aggregate['partition'].items()]+[['合計',len(rows),aggregate['total_delay_sec']]])
    return text


def history_tables(data: dict, rows: list[dict]) -> str:
    """開始判定を同じ176試合で比較し、実際に救える抑止例も数える。"""
    history = data['history']
    names = dict(boundary='連続観測した正式境界', zero3='境界＋5秒以内の得点ゼロ3フレーム',
        stable_zero='境界＋両側STABLEかつ得点ゼロ', empty_zero='境界＋両側空盤面かつ得点ゼロ',
        death_and_zero3='境界＋直前試合の観測死亡＋得点ゼロ3フレーム')
    indexed = {(r['record'],r['game']):r for r in history['rows']}
    values = []
    for name,title in names.items():
        rescued = []
        for row in rows:
            stamp = indexed[(row['record'],row['local_game'])]['alternatives'][name]
            blocked = row['h1_same_time']['first_sec']
            if stamp is not None and blocked is not None and stamp <= blocked:
                rescued.append(row)
        result = history['summary'][name]
        values.append([title,result['reduced'],result['remaining'],len(rescued),
                       sum(r['h1_same_time']['restored_sec'] for r in rescued)])
    text = '開始未観測65/176件の代替判定（元H1とのOR）。時刻tまでの記録だけで判定する。\n\n'
    text += table(['追加する開始証拠','未観測の減少','残る未観測','55件中のH1抑止救済数','同時刻証明で戻る秒'], values)
    text += '勝利数パネルの既存JSONはvideo_c*の区間後の勝者差分で、今回18記録へ結べるオンライン検出時刻がないため数に含めない。match_end原信号も評価入力に保存されていない。境界は収集器が得点リセットを受理してgame_idxを増やした時刻を使う。\n\n'
    text += '176は元HISTORY.jsonと同じ記録内試合の母数で、クリップ間の重複を含む。記録先頭の17例にはこの追加証拠がなく、境界以前の欠落履歴を推定で補っていない。\n\n'
    source_rows = []
    for source in dict.fromkeys(r['record'] for r in history['rows']):
        group = [r for r in history['rows'] if r['record'] == source]
        missing = [r for r in group if not r['old_complete']]
        source_rows.append([source,len(group),len(missing),
            sum(r['alternatives']['zero3'] is None for r in missing),
            sum(r['alternatives']['death_and_zero3'] is None for r in missing)])
    text += table(['記録','試合数','元H1未観測','境界＋ゼロ3後','観測死亡も要求した後'], source_rows)
    return text


def joint_table(data: dict) -> str:
    """単独解除で戻らない例への、二条件解除の追加分を示す。"""
    values = []
    for mode, result in data['joints'].items():
        names = ', '.join(f"{label(r)}@{r['first_sec']:.3f}" for r in result['cases'])
        values.append([mode, result['evaluated_cases'], result['additional_count'],
                       result['additional_sec'], names or 'なし'])
    text = '重複条件の追加再生。該当記録だけを選んだ診断であり、55件全件の複数条件解除ではない。追加件数は全単独解除で未回復だった例だけ。行間の重複は加算できない。\n\n'
    return text + table(['同時に外した条件','再生範囲の対象件数','単独解除に対する追加件数','追加回復秒','試合と初回確定秒'], values)


def outlier_table(data: dict) -> str:
    """重複するzenchi報告を除き、55件と既存q/fcの消失から上位八件を取る。"""
    observations = read(OUT/'OBSERVATIONS_v2.json')
    notes = {(r['group'],r['game']):r for r in read(OUT/'VISUAL_NOTES.json')['cases']}
    tails = {(r['group'],r['game']):r for r in read(OUT/'TAIL_NOTES.json')}
    rows = sorted(data['cases'], key=lambda r:r['delay_sec'], reverse=True)[:8]
    values = []
    for row in rows:
        note = notes[(row['group'],row['game'])]
        key = f"{row['group']}_{row['game']}"
        counts = observations[row['record']]['intervals'][key]
        chain = f"{counts.get('1P_CHAIN',0):.3f} / {counts.get('2P_CHAIN',0):.3f}"
        values.append([label(row),row['side'],row['delay_sec'],chain,note['note']+' '+tails[(row['group'],row['game'])]['note'],
                       f"[実フレーム]({note['contact']})／[末尾1秒刻み](frames/{key}_tail.jpg)"])
    return table(['試合','側','遅れ秒','記録上のCHAIN秒 1P/2P','区間で起きていたこと','根拠画像'], values)


def artifacts(data: dict) -> None:
    """個別入力と全系列を保存し、大きい診断原票の配布版はgzipにする。"""
    exact = read(OUT/'EXACT_INPUTS.json')
    directory = OUT/'cases'
    directory.mkdir(exist_ok=False)
    for row in data['cases']:
        name = f"{row['group']}_{row['game']}"
        case = dict(row, exact_inputs={mode:value[name] for mode,value in exact[row['record']].items()})
        case['recognition_delta_at_original_first'] = recognition_delta(case['exact_inputs'])
        with (directory/f'{name}.json').open('x', encoding='utf-8') as stream:
            json.dump(case, stream, ensure_ascii=False, indent=1)
    with gzip.open(OUT/'ANALYSIS.json.gz', 'xb') as stream:
        stream.write((OUT/'ANALYSIS.json').read_bytes())
    summary = dict(data)
    summary['cases'] = [{k:v for k,v in r.items() if k not in ('inputs','first','original','reason_series')}
                        for r in data['cases']]
    save('SUMMARY.json', summary)


def main() -> None:
    """全表と再現手順を新規保存する。"""
    save('RESOURCE_report.json', resource())
    data = read(OUT/'ANALYSIS.json')
    rows = [r for r in data['cases'] if r['group'] in ('set1','set2')]
    assert len(rows) == 55
    text = 'D-v2d 消失診断（42d82b1、2026-10-02、合否外）\n\n'
    text += '本番コード・設定・既存成果物を変更せず、55件を固定して診断した。既報の「正しい」は最終勝敗との一致であり、その時点の物理的な回避不能性の検証ではない。\n\n'
    text += '元遅れ合計は371.400秒。回復秒は観測死亡秒−max(本番初回秒, 解除再生初回秒)。重複なしの最良単独解除の合計は'
    text += f"{data['aggregate']['union_restored_cases']}件・{data['aggregate']['union_restored_sec']:.3f}秒。修正実装の効果保証ではない。\n\n"
    text += aggregate_tables(data, rows)
    text += joint_table(data)
    text += '55件の全表。M=color_score_mismatch、B=board_unsettled、C=completion_prefix_unverified、I=invalid_bound_input、S=optimistic_survivor、L=bound_limit、U=unverified_attack_or_budget、P=missing_post_counter_completion。E35″表記は未知ツモを前置した呼出し。UだけではD3′の因果を断定せず一つ抜き結果で確認した。\n\n'
    text += '系列は出現順に重複を省いた短縮表示。各試合リンクに、全条件遷移、完全同時刻の旧・新入力、盤面、得点、NEXT、台帳、参照行番号を保存した。\n\n'
    text += '未知手持ち条件はnon_chain=非CHAIN、firing_unplaced=発火設置未確認、next_changed=NEXT不一致、next_motion=NEXT移動、ended=終了観測、end_signal=連鎖終了信号あり。内部入力スナップショットは約1秒間隔で、各実測時刻を保持する。\n\n'
    text += cases_table(rows)
    text += '遅れ上位8件（55件＋既存q/fcの16消失。重複するzenchiの別集計は除外）。CHAIN秒は記録状態の積算で、実映像の連鎖時間の正解値ではない。\n\n'
    text += 'この外れ値比較用71例は、独立検収の「履歴完全試合の本番確定71件」とは別集合。\n\n'
    text += outlier_table(data)
    text += history_tables(data, rows)
    text += '原因別の修正案と危険評価。ここでは案の採否を扱わない。\n\n'
    text += table(['原因','候補箇所 file:line','修正案','誤確定を増やす危険'], FIXES)
    text += '確認範囲：観測用フック付き16記録の表示NPZ全列が既存v2dの同一接頭部と一致。一つ抜きは対象55件を含む14記録を各5構成、先頭履歴を省かず対象の観測死亡直後まで再生した。全計算はrun_limited・nice 10、起動時空き6GiB以上。初回のnice加算は19検出で再生前停止し、再起動済み。\n\n'
    text += 'D4の一つ抜きは既存の旧認識記録への差し替え。拒否した固定セルやplacement_validatedそのものは記録されていないため、そのセル単位の帰属は未確定。各ケースの入力差と回復有無までは実測した。\n'
    with (OUT/'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    artifacts(data)


if __name__ == '__main__':
    main()
