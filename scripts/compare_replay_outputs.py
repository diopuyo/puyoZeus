"""二つの再生出力ディレクトリ (E36b 形式) を全ファイルのバイト単位で比較し、母数付きで報告する。

時間や経過秒を含む DONE/status/audit の elapsed 系は比較対象から外し、除外した項目は必ず表示する。
--ignore-proof-detail: E35 の証明の内訳 (post_counter_bound の proofs と audit の proofs) を比較から外す。
  post_counter_early_exit (既定OFF) は判定を変えず内訳だけを短くするので、その確認にだけ使う。
使い方: python -m scripts.compare_replay_outputs <基準root> <比較root> [--ignore-proof-detail]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PER_SOURCE = ('q_7gc4TgFig', 'review', 'fcXG83vInDY', 'mia8KCjr52g', 'zenchi')
# 出力ファイルのうち、所要時間そのものを含むため比較しないもの (理由付きで表示する)
TIME_DEPENDENT = ('DONE.json', 'status.json', 'RUN.lock')
# audit 系は elapsed_sec などの壁時計を含むので、その項目だけ除いて内容比較する
WALL_CLOCK_KEYS = ('elapsed_sec', 'elapsed_seconds', 'filter_sec', 'enumeration_sec')


def source_dir(root: Path, source: str) -> Path:
    """出典ごとの出力先 (E36b の配置規則)。"""
    on = root / 'on'
    if source in ('review', 'zenchi'):
        return on / source
    return on / 'renders' / source / 'on'


def strip_clock(value: object) -> object:
    """壁時計の項目だけ落とした JSON 値。"""
    if isinstance(value, dict):
        return {k: strip_clock(v) for k, v in value.items() if k not in WALL_CLOCK_KEYS}
    if isinstance(value, list):
        return [strip_clock(v) for v in value]
    return value


PROOF_DETAIL_KEY = 'proofs'


def strip_proofs(value: object) -> object:
    """証明の内訳だけを落とした JSON 値 (判定・理由・候補数などは残す)。"""
    if isinstance(value, dict):
        return {k: strip_proofs(v) for k, v in value.items() if k != PROOF_DETAIL_KEY}
    if isinstance(value, list):
        return [strip_proofs(v) for v in value]
    return value


def digest(path: Path, ignore_proofs: bool = False) -> str:
    """ファイル内容の指紋。JSON は壁時計項目を除いた正規化後。"""
    if path.suffix == '.json':
        try:
            data = strip_clock(json.loads(path.read_text()))
            data = strip_proofs(data) if ignore_proofs else data
            return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        except (ValueError, UnicodeDecodeError):
            pass
    if ignore_proofs and path.name == 'events.jsonl':
        lines = [json.dumps(strip_proofs(json.loads(line)), sort_keys=True)
                 for line in path.read_text().splitlines()]
        return hashlib.sha256(chr(10).join(lines).encode()).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(base: Path, other: Path, ignore_proofs: bool = False) -> dict:
    """ファイル数・一致数・不一致名を返す。"""
    report = dict(files=0, identical=0, different=[], missing=[], skipped=[])
    for src in sorted(base.rglob('*')):
        if not src.is_file():
            continue
        rel = src.relative_to(base)
        if src.name in TIME_DEPENDENT or 'runtime' in rel.parts:
            report['skipped'].append(str(rel))
            continue
        report['files'] += 1
        dst = other / rel
        if not dst.exists():
            report['missing'].append(str(rel))
        elif digest(src, ignore_proofs) == digest(dst, ignore_proofs):
            report['identical'] += 1
        else:
            report['different'].append(str(rel))
    return report


def main() -> None:
    base, other = Path(sys.argv[1]), Path(sys.argv[2])
    ignore_proofs = '--ignore-proof-detail' in sys.argv[3:]
    total = dict(files=0, identical=0)
    for source in PER_SOURCE:
        rep = compare(source_dir(base, source), source_dir(other, source), ignore_proofs)
        total['files'] += rep['files']
        total['identical'] += rep['identical']
        print(source, json.dumps(dict(files=rep['files'], identical=rep['identical'],
              different=rep['different'], missing=rep['missing'], skipped=len(rep['skipped'])),
              ensure_ascii=False))
    print('TOTAL', json.dumps(total), 'ALL_IDENTICAL' if total['files'] == total['identical'] else 'DIFF')


if __name__ == '__main__':
    main()
