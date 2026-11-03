"""補助表示監査・最終構文点検・全表生成を、一つの資源枠内で順に完了する。"""
from __future__ import annotations

from scripts import falsedeath_v2d_display as display
from scripts import falsedeath_v2d_static_checks as checks
from scripts import falsedeath_v2d_report as report


def main() -> None:
    """採点・再生の原票は維持し、報告用の派生表だけを再生成する。"""
    display.main()
    initial = checks.OUT/'STATIC_CHECK_INITIAL.json'
    assert not initial.exists()
    (checks.OUT/'STATIC_CHECK.json').rename(initial)
    checks.main()
    report.main()
    print('補助表示監査・最終構文点検・全表生成が完了')


if __name__ == '__main__':
    main()
