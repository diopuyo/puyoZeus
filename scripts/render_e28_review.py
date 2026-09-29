"""E28で明示されたE27候補承認を記録し、レビュー動画を先に生成する。"""
from __future__ import annotations

from pathlib import Path
import json
from scripts.run_e3_exchange_eval_20260926 import save_json
from scripts.run_e27 import OPTIONS
from scripts.render_e27_review import run


def main() -> None:
    """場面不合格の原票を変えず、ユーザーの新しい採用候補判断を別記録する。"""
    metrics = json.loads(Path('logs/e27/on/METRICS.json').read_text())
    save_json(Path('logs/e28/CANDIDATE_DECISION.json'), dict(candidate='E27',
        source_commit='87b0fca', approved_by='user_explicit_E28_request', options=OPTIONS,
        rationale='全体指標でE22を上回るため新しい採用候補。場面基準は不合格を維持。',
        original_acceptance=metrics, production_config_changed=False))
    run(approved_candidate=True)


if __name__ == '__main__':
    main()
