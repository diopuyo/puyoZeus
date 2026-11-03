"""診断・合否外の短区間原票から全表と再現情報を作る。"""
from __future__ import annotations

import ast
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts._diag_falsedeath_v2b_ablate import DEST, OUT, save
from scripts._diag_falsedeath_v2b_tables import scene
from scripts._diag_set2_falsedeath_audit import events

MODES = ('control', 'A', 'B')
REASONS = ('unverified_attack_or_budget', 'fallback_search',
           'missing_post_counter_completion', 'post_counter_upper_bound')
COLORS = {0: '?', 1: '赤', 2: '青', 3: '緑', 4: '黄', 5: '紫'}
MAX_FUNCTION_LINES = 50
TIMES_PER_ROW = 8
INTRO = """# D-v2b 外し比較 — 診断・合否外

**診断・合否外。採否・合否は判定しない。** reviewの同じ3:00場面だけを対象にした原因帰属診断。
E35′を外したAは直接表示が9.600000秒、従来採点が9.466667秒早まった。
D3′を外したBは両時刻とも元v2bと同じ。この場面の表示遅れはE35′の未知ツモ追加に感応する。
この結果からE35′の撤去や安全性、他場面での採否は判断しない。

作業場所は `D:/puyo_analyzer/wt_evalset`。既存コード・設定・成果物と他worktreeを変更せず、
新規診断スクリプトと `logs/eval_set/falsedeath_fix/v2b_ablate/` の新規出力だけを追加。
前回と同じD4再収集記録 `collect/records/review.jsonl.gz` を先頭2549.05秒から再生し、2775.2秒で停止。
表示開始2579.05秒、場面抽出2750〜2775秒、閾値p1≥0.95、従来採点は同じEMA再適用。
門2766.0秒は前回の文脈としてのみ記載し、今回の合否判定には使用しない。全体再生・全体採点は実施しない。

## 構成と表示時刻

D1とD4は全構成ON、D2/D2′はOFF。AはD3′ ON・E35′ OFF、BはD3′ OFF・E35′ ON。
BのD3（旧版）もOFF。実引数は各構成のCOMMAND.json。
元v2bは観測対照として追加再生し、元成果物との表示・イベントのバイト一致を検証した。

"""
DETAIL = """
## 同一入力の証明対比（2P、2760〜2771秒、両端を含む）

各実呼出しの盤面候補・受け量・手数・色集合・経過時間・前提条件を共通にし、
キューを `(0,0,NEXT1,NEXT2)` と `(NEXT1,NEXT2)` にした二通りを独立キャッシュで計算。
反実仮想を表示やtrackerに戻していない。実構成と同じ条件の再計算結果は全257呼出しで一致。
探索前に拒否された呼出しは「予算未確認」「完走前提なし」とし、探索不成立と区別する。

元v2bの59呼出し（2761.016667〜2768.350000秒）は未知なしなら全て死亡証明成立。
全て認識状態CHAIN、NEXT1=黄紫、NEXT2=赤紫。未知ありでは全て `bound_limit`、2049ノード
（NODE_LIMIT=2048超過）だった。これは生存枝発見ではなく計算予算による証明打切り。
Bの未知あり56呼出しも全て2049ノードで打切り。Bの2760.783333秒はNEXT欠測で、未知なしでも打切り。
Bでは2760.116667秒の同一入力・未知なし証明が成立するが、両フラグを外した表示再生は行っていない。

"""
STATES = """
## 全呼出し時点の状態・NEXT・証明結果

下表は257呼出し全て（元v2b=86、A=86、B=85）。同一状態・NEXT・結果の時点だけを併記し、
時刻を省略していない。表にない時刻・構成には対象E35呼出しがない。秒は小数6桁に丸め、原票は元精度。
`bound_limit:1` は候補1枚の証明が探索上限で打切り。成立は候補全ての死亡証明成立を表す。

「自連鎖中」は認識状態CHAIN、「終了待ち含む連鎖中」はE35が使う `_chaining`。
「発火手設置済」はCHAINから読める発火手の状態であり、次の操作ツモまで設置済という意味ではない。
GRAVITY_SETTLEは連鎖後の重力整列中。STABLEは確定盤面であり、手持ちツモの有無・設置完了の直接証拠ではない。
今回の構成には配置観測器 `hands_observation` がなく、独立した置き完了フラグは取得できない。
2Pの `post_end_drop_sec` も全呼出しで欠測。この限界を補間せず、認識から読み取れる状態だけを表示する。
2770.183333秒以降は物理STABLEでも終了待ちが残り、2770.616667秒で `_chaining=False` に変わった。
NEXTは記録された色で、映像真値として再確認した色ではない。

"""


def read(path: Path) -> Any:
    """原票のUTF-8を読む。"""
    return json.loads(path.read_text(encoding='utf-8'))


def proof_label(value: dict) -> str:
    """前提未達・探索打切り・生存枝・死亡証明を区別する。"""
    if value['dead']:
        return '成立'
    if value['reason'] != 'fallback_search':
        return {'unverified_attack_or_budget': '予算未確認',
                'missing_post_counter_completion': '完走前提なし'}.get(value['reason'], value['reason'])
    reasons = Counter(p['reason'] for p in value['proofs'])
    return ','.join(f'{key}:{count}' for key, count in sorted(reasons.items()))


def color(pair: list | None) -> str:
    """軸・子の記録順を保つ。"""
    return '欠測' if pair is None else ''.join(COLORS[v] for v in pair)


def call_cells(row: dict) -> list[str]:
    """呼出しごとの状態と二通りの証明を表へ変換する。"""
    yes = lambda value: '有' if value else '無'
    return [row['state'], yes(row['physical_chain']), yes(row['chaining']),
            row['placement_state'], color(row['next1']), color(row['next2']),
            proof_label(row['with_unknown']), proof_label(row['without_unknown'])]


def table(headers: list[str], rows: list[list]) -> str:
    """指定されたMarkdown全表を生成する。"""
    lines = ['|' + '|'.join(headers) + '|', '|' + '|'.join('---' for _ in headers) + '|']
    lines += ['|' + '|'.join(map(str, row)) + '|' for row in rows]
    return '\n'.join(lines) + '\n'


def summaries() -> dict:
    """保存イベント数と実呼出し数の分母を別々に集計する。"""
    result = {}
    for mode in MODES:
        directory = DEST / mode
        calls = read(directory / 'calls.json')
        reasons = Counter(p['reason'] for event in events(directory / 'events.jsonl')
                          for v in event['values'] if 2760 <= v['t_sec'] <= 2772
                          for p in v.get('post_counter_bound', ()))
        pairs = Counter((proof_label(r['with_unknown']), proof_label(r['without_unknown'])) for r in calls)
        result[mode] = dict(scene=scene(directory / 'display.npz'), reasons=dict(reasons),
            saved_n=sum(reasons.values()), call_n=len(calls), status=read(directory / 'status.json'),
            comparisons=[dict(with_unknown=a, without_unknown=b, count=n) for (a, b), n in pairs.items()])
    return result


def call_table() -> str:
    """同一時刻・同一結果の構成だけをまとめ、全呼出しを欠落なく載せる。"""
    merged: dict[tuple, list[str]] = {}
    count = 0
    for mode in MODES:
        for row in read(DEST / mode / 'calls.json'):
            key = (row['t_sec'], *call_cells(row))
            merged.setdefault(key, []).append('v2b' if mode == 'control' else mode)
            count += 1
    assert sum(map(len, merged.values())) == count
    groups: list[tuple[list[str], tuple]] = []
    for key, modes in sorted(merged.items()):
        cells = ('/'.join(modes), *key[1:])
        if groups and groups[-1][1] == cells and len(groups[-1][0]) < TIMES_PER_ROW:
            groups[-1][0].append(f'{key[0]:.6f}')
        else:
            groups.append(([f'{key[0]:.6f}'], cells))
    rows = [[', '.join(times), *cells] for times, cells in groups]
    return table(['全呼出し秒（同一状態・結果のみ併記）', '構成', '物理状態', '自連鎖中', '終了待ち含む連鎖中',
                  '置き完了状態（認識上）', 'NEXT1', 'NEXT2', '未知あり', '未知なし'], rows)


def verify_sources() -> dict:
    """新規スクリプトの構文・関数長と原票のハッシュを記録する。"""
    root = Path(__file__).resolve().parents[1]
    scripts = [root / 'scripts' / name for name in ('_diag_falsedeath_v2b_ablate.py',
               '_diag_falsedeath_v2b_ablate_report.py')]
    lengths = {}
    for path in scripts:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                assert length <= MAX_FUNCTION_LINES, (path.name, node.name, length)
                lengths[f'{path.name}:{node.name}'] = length
    inputs = [OUT / 'collect/records/review.jsonl.gz', OUT / 'v2b/short/review/display.npz',
              OUT / 'v2b/short/review/events.jsonl', root / 'src/production_config.py']
    return dict(function_lines=lengths, sha256={str(p.relative_to(root)):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                control_equivalence=read(DEST / 'control/equivalence.json'))


def summary_tables(result: dict) -> str:
    """時刻・保存理由・証明結果の全構成を並べる。"""
    names = {'control': '元v2b（観測対照）', 'A': 'A：E35′なし', 'B': 'B：D3′なし'}
    timing = [[names[m], f"{result[m]['scene']['direct_sec']:.6f}",
               f"{result[m]['scene']['legacy_sec']:.6f}", result[m]['scene']['source']] for m in MODES]
    text = table(['構成', '直接表示p1≥0.95（秒）', '従来採点（秒）', '最初の表示経路'], timing)
    text += '\n## E35保存理由（2760〜2772秒、前回と同形式）\n\n'
    reasons = [[reason, *[f"{result[m]['reasons'].get(reason, 0)} / {result[m]['saved_n']}"
                          for m in MODES]] for reason in REASONS]
    text += table(['理由', '元v2b 件数／母数', 'A 件数／母数', 'B 件数／母数'], reasons)
    text += '\n母数は保存されたpost_counter_boundの値数。独立試合数・実呼出し総数ではない。\n'
    text += DETAIL
    outcomes = [('予算未確認', '予算未確認'), ('bound_limit:1', '成立'),
                ('bound_limit:1', 'bound_limit:1'), ('完走前提なし', '完走前提なし'), ('成立', '成立')]
    rows = []
    for a, b in outcomes:
        counts = [sum(v['count'] for v in result[m]['comparisons']
                      if (v['with_unknown'], v['without_unknown']) == (a, b)) for m in MODES]
        rows.append([a, b, *counts])
    rows.append(['合計', '実呼出し数', *[result[m]['call_n'] for m in MODES]])
    return text + table(['未知あり', '未知なし', '元v2b', 'A', 'B'], rows)


def execution_text(result: dict) -> str:
    """再生の母数と観測の非干渉を記述する。"""
    rows = [[m, result[m]['status']['frames'], result[m]['status']['display_frames'],
             f"{result[m]['status']['elapsed_seconds']:.3f}",
             f"{read(DEST / m / 'resources.json')['memory_available_kib'] / 1024**2:.3f}"] for m in MODES]
    text = '\n## 実行・検証\n\n'
    text += table(['構成', '処理行', '表示行', '再生本体（秒）', '起動時空きメモリ（GiB）'], rows)
    text += """
共通run_limited、nice10、BLAS各1、実計算上限6本、空き6GiB以上で逐次実行。
前回同様、確認済み制御2モジュールだけをワーカー数から除外。他ジョブは変更していない。
元v2bのdisplay.npz/events.jsonlはバイト一致、events.diagnostics.jsonは内容一致。
新規Pythonの構文・全関数50行以内、全257呼出しの同条件証明一致と全呼出し表の件数保存を確認。

初回Aは診断側が存在しない配置観測器を参照して停止。failed_A_attempt1/と同名ログに失敗記録を保持し、
診断側だけを欠測対応してA/B/対照を再生し直した。失敗実行は上表に含めない。既存コードの変更はない。

SUMMARY.jsonは集計原票、CALL_TABLE.mdは全呼出し表、VERIFICATION.jsonは一致検証・関数長・入力ハッシュ。
各構成のcalls.jsonは入力・盤面候補・状態・双方の証明を全て保持し、圧縮版calls.json.gzもコミットする。
表示NPZとevents.jsonl.gz、COMMAND/status/resources、監査圧縮版も保存する。
元の非圧縮再生成果物は同じ新規出力先に残す。再現手順は新規run_falsedeath_v2b_ablate.sh、集計は
`python -B -m scripts._diag_falsedeath_v2b_ablate_report`、報告・圧縮は同モジュールの `--package`。
全処理は完成済み出力の上書きを拒否するため、再実行時は新しい出力先が必要。

コミット署名は前回と同じCo-Authored-By / Claude-Sessionの2行。**全結果は診断・合否外。**
"""
    return text


def package() -> None:
    """報告と圧縮原票を新規作成し、解凍時のバイト一致を検査する。"""
    result = read(DEST / 'SUMMARY.json')
    report = INTRO + summary_tables(result) + STATES + (DEST / 'CALL_TABLE.md').read_text(encoding='utf-8')
    report += execution_text(result)
    with (DEST / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write(report)
    for mode in MODES:
        for name in ('calls.json', 'events.jsonl', 'post_counter_bound_audit.json'):
            path = DEST / mode / name
            raw = path.read_bytes()
            packed = gzip.compress(raw, mtime=0)
            assert gzip.decompress(packed) == raw
            with path.with_suffix(path.suffix + '.gz').open('xb') as stream:
                stream.write(packed)
    save(DEST / 'PACKAGE_VERIFICATION.json', verify_sources())


def main() -> None:
    """集計と全呼出し表を新規出力する。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', action='store_true')
    if parser.parse_args().package:
        package()
        return
    result = summaries()
    save(DEST / 'SUMMARY.json', result)
    save(DEST / 'VERIFICATION.json', verify_sources())
    with (DEST / 'CALL_TABLE.md').open('x', encoding='utf-8') as stream:
        stream.write(call_table())
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
