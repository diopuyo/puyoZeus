"""
milestones.jsonl を data/milestone_last_seen 以降でスキャンし、
通知すべきイベントを標準出力に JSON Lines で吐く。

通知対象:
    - new_best かつ kind_detail == "global_best" (holdout 更新)
    - cycle_complete
    - fatal
    - anomaly
    - early_exit
    - phase_complete (phase3 のみ = 動画 DL 完了)

使い方:
    ./venv/bin/python scripts/check_milestones_for_notify.py [--commit]

--commit を付けると data/milestone_last_seen を最新行に更新する。
付けない場合は読み取りのみ（dry-run）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAST_SEEN = Path("data/milestone_last_seen")
MILESTONES = Path("data/milestones.jsonl")

SIMPLE_NOTIFY_KINDS = {"cycle_complete", "fatal", "anomaly", "early_exit"}


def is_notify(entry: dict) -> bool:
    kind = entry.get("kind", "")
    if kind == "new_best" and entry.get("kind_detail") == "global_best":
        return True
    if kind in SIMPLE_NOTIFY_KINDS:
        return True
    if kind == "phase_complete" and entry.get("phase") == "phase3":
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true", help="last_seen を更新する")
    args = parser.parse_args()

    last = int(LAST_SEEN.read_text().strip()) if LAST_SEEN.exists() else 0
    lines = MILESTONES.read_text(encoding="utf-8").splitlines() if MILESTONES.exists() else []
    new_lines = lines[last:]

    notif = []
    for ln in new_lines:
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if is_notify(e):
            notif.append(e)

    # 読みやすいサマリを stderr へ、JSON Lines を stdout へ
    sys.stderr.write(f"last_seen={last}  total={len(lines)}  new={len(new_lines)}  notify={len(notif)}\n")
    for e in notif:
        summary = (e.get("summary") or "")[:120]
        sys.stderr.write(f"  {e.get('ts')} {e.get('kind')}: {summary}\n")
        sys.stdout.write(json.dumps(e, ensure_ascii=False) + "\n")

    if args.commit:
        LAST_SEEN.write_text(str(len(lines)))
        sys.stderr.write(f"last_seen を {len(lines)} に更新\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
