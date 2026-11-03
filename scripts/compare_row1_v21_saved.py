"""v2.1保存画素再生の内部履歴・B3入力・特徴列・表示を照合する。"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_row1_v2_saved import compare, read

COMPARE_KEYS = ('internal', 'prev_confirmed', 'evaluation', 'counts',
                'b3_input', 'features', 'display', 'raw')


def baseline_result(blob: bytes) -> bytes:
    """追加した未使用Noneフィールドのみ除き、旧OFFの全結果をバイト照合する。"""
    result = pickle.loads(blob)
    for side in (result.p1, result.p2):
        assert side.evaluation_confirmed_board is None
        side.__dict__.pop('evaluation_confirmed_board', None)
    return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    off, on, base = [read(Path(p)) for p in sys.argv[1:4]]
    assert [r['frame'] for r in off] == [r['frame'] for r in on] == [r['frame'] for r in base]
    result = dict(scope='PC1保存実画素128コマ・10Hz冷起動。elのf70734再走ではない。',
        frames=len(off), mismatches={k: sum(a[k] != b[k] for a, b in zip(off, on))
                                    for k in COMPARE_KEYS}, public=compare(off, on),
        baseline_off=dict(result_ignoring_added_none=sum(baseline_result(a['result']) != b['result']
                           for a, b in zip(off, base)),
                          display=sum(a['display'] != b['display'] for a, b in zip(off, base))))
    Path(sys.argv[4]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    assert not any(result['mismatches'].values())
    assert not any(result['baseline_off'].values())
    assert result['public']['outside_cells'] == 0


if __name__ == '__main__':
    main()
