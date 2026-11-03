"""途中入力付き記録でOFFとE25包括条件の既存出力を厳密照合する。"""
from __future__ import annotations

from pathlib import Path
from scripts.replay_exchange_event_20260926 import replay, compare
from scripts.run_e22_death_formula_20260928 import OPTIONS
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e26')


def main() -> None:
    """5記録の入力不変検証と組み合わせ、レビューとqで無効時の出力を確認する。"""
    results = {}
    cases = [('off_review', 'review', Path('logs/e22/on/review'), OPTIONS),
        ('off_q', 'q_7gc4TgFig', Path('logs/e22/on/renders/q_7gc4TgFig/on'), OPTIONS),
        ('legacy_e25', 'review', Path('logs/e25/on/review'),
         dict(OPTIONS, multi_landing_death=True, landing_state_safety=True))]
    for name, source, baseline, options in cases:
        dest = OUT/'controls'/name
        dest.mkdir(parents=True, exist_ok=True)
        replay(OUT/'records'/f'{source}.jsonl.gz', dest, Path('models/exchange_event_v3'), True, **options)
        results[name] = compare(baseline, dest)
        save_json(OUT/'CONTROLS.json', results)
        print(name, results[name], flush=True)


if __name__ == '__main__':
    main()
