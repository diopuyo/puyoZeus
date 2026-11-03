"""追加計装の最大通知と、同じ入力の親プロセス特徴計算を対応させる。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_rt_integration_20261002 import rows

ROOT = Path('logs/rt_review_20261002/detail/audit')


def main() -> None:
    notices = rows(ROOT, 'notifications-*.jsonl')
    profiles = rows(ROOT, 'static_profile-*.jsonl')
    slowest = max(notices, key=lambda r: r['milliseconds'])
    feature = max(profiles, key=lambda r: r['milliseconds'])
    result = dict(profiling_overhead_included=True, slowest=slowest,
                  longest_static_features=feature)
    Path('docs/verification/rt_review_profile_20261002.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
