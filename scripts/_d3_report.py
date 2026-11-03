"""D3の観測母数・未確定・遅延下限を区別した報告書を生成する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from scripts._d3_inventory import OUT, SOURCES

REPORT = Path('docs/D3_CEILING_EARLY_CONFIRM_20260929.md')
LABELS = {'NEXT': 'NEXT実移動', 'formula': '掛け算式', 'ojama': 'おじゃま流入'}


def number(value: Any) -> str:
    """欠測をゼロへ変換せず表示する。"""
    return '未測定' if value is None else f'{value:.2f}'.rstrip('0').rstrip('.')


def counts(label: str, data: dict) -> str:
    """合図周期と、置きに対応できた母数を分けた行を作る。"""
    fields = ('signals', 'physical_placements', 'paired', 'early', 'comparison_events', 'comparison_cells', 'error_events', 'error_cells', 'persistent_cells', 'unresolved')
    return '| '+label+' | '+' | '.join(str(data[key]) for key in fields)+' |'


def count_table(title: str, entries: list[tuple]) -> list[str]:
    """診断表の母数を省略しない。"""
    return [f'## {title}', '',
        '| 区分 | 観測合図周期 | 画像で置き確認 | 書込み対応数 | 早期確定 | 初回盤面差分の置き | 初回差分セル | 実際の誤りを観測した置き | 実観測誤りセル | 合図時残存セル | 書込み対応未確定 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        *[counts(label, data) for label, data in entries], '']


def latency_table(events: list[dict]) -> list[str]:
    """元動画FPS別の追加待ちと、8フレーム超過の下限件数を示す。"""
    lines = ['## 合図待ちの追加遅延（原動画フレーム）', '',
        '| 着地段 | FPS | 早期確定母数 | P50 | P95 | 最大 | 追加待ちだけで8超 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for stage in range(1, 13):
        for fps in (30, 60):
            group = [e for e in events if e['status'] == '対応候補' and e['early'] and e['stage'] == stage and e['fps'] == fps]
            values = [e['added_delay_native_frames'] for e in group]
            if not values:
                continue
            lines.append('| '+ ' | '.join(map(str, [stage, fps, len(values),
                number(float(np.percentile(values, 50))), number(float(np.percentile(values, 95))),
                max(values), sum(v > 8 for v in values)]))+' |')
    return lines+['', '追加待ち = max(0, 最初の観測合図 − 最初の対応STABLE書込み)。この観測合図まで待つ条件では、書込み時に画像上の接地を要求しているため総遅延はこの値以上。未検出の先行合図がある場合の理想検出器の合否は断定しない。追加待ち8以下だけでも総遅延8以内を認定しない。NEXT開始は絵柄移動またはNEXT列の繰上りで裏付けた画素変化開始の推定で、`onset_uncertainty_frames`に移動確証までの幅を保存した。', '']


def overlap_text(overlap: list[dict]) -> list[str]:
    """直接対応したD1セルだけを列挙し、未照合をゼロ誤りとしない。"""
    audit = json.loads((OUT/'d1_write_audit.json').read_text())
    columns = ('source', 'game', 'side', 'trigger', 'row', 'col')
    by_cell = {tuple(c[k] for k in columns): c['direct_matches'] for c in audit}
    for cell in overlap:
        cell['write_audit_matches'] = by_cell[tuple(cell[k] for k in columns)]
    matched = [c for c in overlap if c['matches'] or c['write_audit_matches']]
    (OUT/'d1_overlap_union.json').write_text(json.dumps(overlap, ensure_ascii=False, indent=2))
    lines = ['## D1の経路未確定96セルとの重なり', '',
        f'同一記録・side・座標・起点色・基準色が一致し、合図からD1発火まで保存値が継続した照合候補は **{len(matched)}/96セル**。',
        'このうち画像基準の置き・書込み対応まで成立したものは '+str(sum(any(m['status'] == '対応候補' for m in c['matches']) for c in matched))+'/96セル。別途、置き対応の成否とは独立に、D1の最終STABLE書換え時の色支持と同一組周期での画像変化も照合した。残りはこの診断で帰属を確定していない。', '']
    if matched:
        lines += ['| 動画 | side | D1発火秒 | セル | 保存→基準 | 対応書込み秒 |', '|---|---:|---:|---|---|---:|']
        for cell in matched:
            stamps = ', '.join(number(v) for v in sorted({m['write_t'] for m in cell['matches']+cell['write_audit_matches']}))
            lines.append(f"| {cell['source']} | {cell['side']+1} | {number(cell['trigger'])} | ({cell['row']},{cell['col']}) | {cell['origin']}→{cell['snapshot']} | {stamps} |")
    return lines+['']


def methods() -> list[str]:
    """原票と代理基準の意味を具体的に定義する。"""
    return ['## 計数方法', '',
        '1. 元記録は`logs/e31/records/{source}.jsonl.gz`。保存盤面・状態・時刻を読み取り、sideごとのSTABLEで前フレームから盤面が変化した更新を抽出。再認識で元の保存盤面を置き換えていない。',
        '2. NEXTは原動画の絵柄の平行移動（テンプレートNCC≥0.80、静止位置比の改善≥0.08、縦移動≥2px）で確認し、直前8原フレームの画素差から開始を推定。速い30fps画像で対応が飛ぶ場合は、3回以上一致した旧DNEXT→新NEXTの繰上りでも補完する。発光だけの`next_slide_motion`は置き完了として数えない。式は既存×テンプレートと閾値0.70を使用し、固定列の上下16px・左右8pxを含めて全原フレームを探索。認識コードは変更していない。',
        '3. NEXTから次のNEXTまでを一つの合図周期とし、その中の最初のNEXT・式出現・おじゃま上端流入を採用。おじゃまはCNN単独=HSV単独の新規9に加え、後続0.5秒内の複数観測で同列の下降、または0.15秒以上の残存を要求し、状態名・単発発光だけの候補を除外。原動画から検出できていない置きの総数は、この周期数からは保証しない。',
        '4. 比較基準は同一パッチのCNN単独とHSV単独の一致。CNNは`models/cnn_phase_b_large_v2.pt`、HSVは既存`ColorClassifier()`。融合器・履歴補完・物理補正を通さず、非一致セルと隠し段row0は未観測。式の場合は出現の1〜4原フレーム前のうち、一致色ぷよが最多のフレームを選び、同数なら一致セル数・新しさで決める。',
        '5. 直前NEXT時点の独立画像と合図時の独立画像を比較し、新規色ぷよ1〜2セルの接地を置きとして確認する。誤った保存盤面を着地前の基準には使わない。この画像基準で空だった場所へSTABLEが書き込んだ色を追跡し、書込み時にもCNN=HSVの色支持と接地を要求する。2セルとも見える場合は組の色内訳も照合。同じ書込みが2周期へ対応する競合は未確定へ戻す。着地段は画像上の新規色ぷよの最上段（row1=天井から1段目）。上端の隠れた一方は直接計数できない。',
        '6. 早期確定は書込み時刻<最初の合図。初回保存との差分数は`initial_comparison_cells`に保存。件数表の誤りはさらに、合図までの実観測でCNN=HSVの基準色に対して保存値が異なったことを要求する。帰属する範囲は同じ組の書込み旧位置と合図時の新位置で、無関係な既存誤り・おじゃま追加は除外。同じ置きで取り込まれなかった色セルは含める。したがって単純な盤面差分数そのものではなく、画像による帰属が成立した下限件数である。',
        '7. 持続時間は最初に誤りを観測した時刻から保存値が基準色へ訂正されるまでの下限。次の置き合図・連鎖・重力・試合終了・記録末尾で打ち切り、基準盤面が変わった後まで同じ色を正解とみなさない。訂正完了分の分位点からは打切りを除く。全誤りの無条件P50/P95としては扱わない。', '']


def main() -> None:
    """数表と追跡可能な原票パスを、指定Markdownへ保存する。"""
    os.nice(19)
    summary = json.loads((OUT/'summary.json').read_text())
    events = json.loads((OUT/'measurements.json').read_text())
    overlap = json.loads((OUT/'d1_overlap.json').read_text())
    total = summary['total']
    lines = ['# D3 天井付近の早期確定診断（2026-09-29）', '',
        '**全5記録・指定全区間の全置きに対する確定集計は未達。** review原票は2777.283333秒で終了しており、指定終端3427.2秒まで649.916667秒の保存盤面がない。原動画があっても元の書込み時刻は補えない。',
        f"記録がある区間で観測した合図周期は{total['signals']}、独立画像で置きを確認した数は{total['physical_placements']}、STABLE書込みと対応した母数は{total['paired']}。以下の早期・差分件数はこの書込み対応母数での計数であり、未対応{total['unresolved']}周期を正常扱いしない。", '',
        '対象はq_7gc4TgFig / fcXG83vInDY / mia8KCjr52gの0〜900秒、zenchiの2580.6〜3427.116667秒、reviewの2580.6〜2777.283333秒。zenchiとreviewは同じ映像でも異なる保存盤面として数え、画像観測だけを共有した。', '']
    lines += count_table('記録別', [(s, summary['sources'][s]) for s in SOURCES]+[('合計', total)])
    lines += count_table('着地段別（天井から1、2、3、…）', [(str(r), summary['stages'][str(r)]) for r in range(1, 13)]+[('未確定', summary['unknown_stage'])])
    lines += count_table('最初の合図別', [(LABELS[s], summary['signals'][s]) for s in LABELS])
    lines += ['初回盤面差分は、最初の早期書込みと合図時の一致色を組の旧・新位置で比較した件数。実観測誤りは、その間に保存値が画像の基準色と違ったことまで確認できた部分集合。', '',
        'D1の既知場面（review/zenchi 2P、2750.300秒）は、直前NEXT画像との差がCNN=HSVで新規5色セルとなり、2セルの組として照合不成立。次の2750.833333秒も前周期画像の非一致により新規追加を確定できず、この2置き・既知4セルは全数表の誤り件数へ加えていない。人手の正解で穴埋めせず、未確定として残した。', '',
        '## 誤りの持続時間', '',
        f"誤り{total['error_cells']}セル全体の観測下限はP50 **{number(total['duration_lower_p50'])}秒**、P95 **{number(total['duration_lower_p95'])}秒**。打切りを含むため真の持続時間分布の分位点ではない。",
        f"訂正まで追跡できた置き{total['event_closed_n']}件の持続時間下限: P50 **{number(total['event_duration_p50'])}秒**、P95 **{number(total['event_duration_p95'])}秒**。",
        f"セル単位では訂正完了{total['closed_n']}セル、P50 {number(total['duration_p50'])}秒、P95 {number(total['duration_p95'])}秒。打切り{total['censored']}セルは別計上。", '']
    lines += latency_table(events)+overlap_text(overlap)+methods()
    lines += ['## 出力', '',
        '- `logs/d3/measurements.json`: 全合図周期、書込み時刻、合図時刻、着地段、誤りセル、継続時間、対応不能理由。',
        '- `logs/d3/summary.json`: 記録別・段別・合図別の母数付き集計。',
        '- `logs/d3/stage_signal_counts.tsv`: 段×合図のクロス集計（未確定段を含む）。',
        '- `logs/d3/d1_overlap_union.json` / `d1_write_audit.json`: D1の96セル全部の対応候補と書換え照合。',
        '- `logs/d3/*_image_signals.json`: 全原フレーム由来の式・NEXT移動と開始時刻の推定幅。',
        '- `logs/d3/*_observations.jsonl`: 画像フレーム番号とCNN単独・HSV単独の72可視セル。',
        '- `scripts/_d3_*.py`: 診断・集計の再現スクリプト。全重処理はnice 19・並列1。',
        '- 確定済み候補・画像観測からの再集計順: `_d3_measure` → `_d3_d1_audit` → `_d3_report`。',
        '', '認識・評価コード、採用フラグ、E34cのファイルは変更していない。コミットは作成していない。', '']
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    print(REPORT, flush=True)


if __name__ == '__main__':
    main()
