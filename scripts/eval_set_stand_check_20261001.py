"""台の確認: 評価セットの再生器が既知の出力を再現するか (display.npz / events.jsonl のバイト一致、2026-10-01)。

1. 再判定再生器 J_orig_rs1 の第3パート = wt_nextfix の既存再生 (q .517927 を出した出力)
2. J_orig_rs0 の第3パート = wt_nextfix R0 (現本番再生。「rs0 は R0 と一致」の主張の確認)
3. 再生CLI 本番指定の第3パート = wt_switch cli_prod (本番配線の検収出力)
使い方: python -m scripts.eval_set_stand_check_20261001
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

E = Path('logs/eval_set')
NF = Path('/mnt/d/puyo_analyzer/wt_nextfix/logs/next_shift_train/replay')
SW = Path('/mnt/d/puyo_analyzer/wt_switch/logs/switch_smoothing/cli_prod/zenchi')
PAIRS = {
    'rejudge_J_orig_rs1': (E/'replay_rejudge/J_orig_rs1/on/zenchi', NF/'J_orig_rs1/on/zenchi'),
    'rejudge_J_orig_rs0_vs_R0': (E/'replay_rejudge/J_orig_rs0/on/zenchi', NF/'R0/on/zenchi'),
    'cli_prod': (E/'replay_cli/prod/zenchi', SW),
}
FILES = ('display.npz', 'events.jsonl')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """各対の各ファイルの SHA を比べて STAND_CHECK.json に保存する。"""
    result = {}
    for name, (mine, known) in PAIRS.items():
        result[name] = {f: dict(identical=sha(mine/f) == sha(known/f), mine=str(mine/f), known=str(known/f))
                        for f in FILES}
    result['passed'] = all(v['identical'] for r in result.values() for v in r.values())
    (E/'STAND_CHECK.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps({k: {f: v['identical'] for f, v in r.items()} if isinstance(r, dict) else r
                      for k, r in result.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
