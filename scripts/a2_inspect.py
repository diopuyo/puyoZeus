"""#6の式通知と連鎖境界を保存入力から抽出する。"""
from pathlib import Path
import json
from src.exchange_event_record import read_records
from src.exchange_formula_origin import FormulaOrigin


def main() -> None:
    """変化点だけを出力し、盤面画像は扱わない。"""
    path = Path('/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/falsedeath_fix/collect/records/s1.jsonl.gz')
    previous = None
    origin = FormulaOrigin()
    for row in read_records(path):
        if row['kind'] != 'update':
            continue
        result, _, final, stamp, game, formula, display, visible = row['args']
        if stamp > 4275:
            break
        if stamp < 4258:
            continue
        side = result.p1
        event = side.chain_event
        corrected, local_total = origin.observe(side, formula[0], display[0], visible[0], stamp)
        detail = dict(state=side.state.name, event=None if event is None else {
            k: getattr(event, k) for k in ('trigger_sec', 'mechanism', 'chain_count', 'total_score')},
            formula=formula[0], display=display[0], score=side.score, visible=visible[0], final=vars(final),
            corrected_count=None if corrected.chain_event is None else corrected.chain_event.chain_count,
            corrected_total=local_total, offset=origin.offset)
        if detail != previous:
            print(json.dumps(dict(t=stamp, game=game, **detail)), flush=True)
        previous = detail
    print(json.dumps(dict(origin_evidence=origin.audit)), flush=True)


if __name__ == '__main__':
    main()
