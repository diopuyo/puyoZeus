"""全体の試験を基準点と今の先端で回したとき、失敗の集合が同じかを比べる (読み取りのみ、重い処理ではない)。

使い方: 各 worktree で `python -m pytest tests/ -q -rfE -p no:cacheprovider > 出力.txt` を回し、2 つの出力を渡す。
  python packaging/compare_pytest_failures.py --base base.txt --head head.txt
基準点にも先端にもある失敗は「既知」、先端にだけある失敗が「新しい失敗」(1 件でもあれば終了コード 1)、
基準点にだけある失敗は「直った失敗」。件数の母数 (passed / failed / error / skipped) も併記する。
注意: 試験の ID は worktree のパスに依存しないので、別の worktree の出力どうしを比べられる。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

OUTCOME = re.compile(r'^(FAILED|ERROR) (\S+)')
SUMMARY_PART = re.compile(r'(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)')
SINGULAR = {'errors': 'error', 'warnings': 'warning'}


def failed_ids(text: str) -> set[str]:
    """short test summary の FAILED / ERROR 行から試験 ID を集める (理由の文は除く)。"""
    found = (OUTCOME.match(line) for line in text.splitlines())
    return {f'{m.group(1)} {m.group(2)}' for m in found if m}


def summary_counts(text: str) -> dict[str, int]:
    """最後の「N passed, M failed ... in 12.3s」行の件数。無ければ空 (出力が途中で切れた = 比較に使わない)。"""
    for line in reversed(text.splitlines()):
        line = line.strip('= ').strip()
        parts = SUMMARY_PART.findall(line)
        if parts and ' in ' in line:
            counts: dict[str, int] = {}
            for number, name in parts:
                key = SINGULAR.get(name, name)
                counts[key] = counts.get(key, 0)+int(number)
            return counts
    return {}


def compare(base_text: str, head_text: str) -> dict:
    """基準点 (base) と先端 (head) の失敗の集合の比較。summary が読めない出力は不合格にする (fail-silent を避ける)。"""
    base, head = failed_ids(base_text), failed_ids(head_text)
    counts = dict(base=summary_counts(base_text), head=summary_counts(head_text))
    complete = bool(counts['base']) and bool(counts['head'])
    return dict(base_failures=len(base), head_failures=len(head), common=len(base & head),
                new_failures=sorted(head-base), fixed_failures=sorted(base-head), counts=counts,
                complete=complete, passed=complete and not (head-base))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--base', type=Path, required=True, help='基準点 (例 278aa36f) の pytest 出力')
    parser.add_argument('--head', type=Path, required=True, help='今の先端の pytest 出力')
    args = parser.parse_args()
    result = compare(args.base.read_text(encoding='utf-8', errors='replace'),
                     args.head.read_text(encoding='utf-8', errors='replace'))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
