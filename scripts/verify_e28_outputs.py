"""E28診断の再現性・制約・成果物を検査する。"""
from __future__ import annotations
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT = Path('logs/e28')
PRODUCTION_SHA = '4152435394271234e9b86ee206cfbe395d7d03ee76b536865ea9ba566e4070b4'


def verify_live() -> None:
    """納品動画の死亡経路と固定E27記録を、入力から照合する。"""
    original = json.loads((OUT/'SCENE_RAW.json').read_text())
    live = json.loads((OUT/'live/SCENE_RAW.json').read_text())
    keys = ('t_sec', 'pending', 'midchain_status', 'hidden_status', 'safety')
    assert len(original) == len(live)
    for left, right in zip(original, live):
        assert all(left[k] == right[k] for k in keys), left['t_sec']
        assert left['last']['multi_landing'] == right['last']['multi_landing'], left['t_sec']
    sample = next(r for r in live if 2755 <= r['t_sec'] < 2755.05)
    save_json(OUT/'LIVE_COMPARISON.json', dict(observations=len(live),
        matched_fields=list(keys)+['multi_landing'], result='identical',
        example_sec=sample['t_sec'], probability_incoming=sample['last']['incoming'][1],
        death_incoming=sample['last']['death_incoming'][1],
        note='489対490は確率入力と死亡専用台帳の区別。固定記録と動画の差ではない。'))


def main() -> None:
    """判定を変えるテストを追加せず、診断と固定入力の一致を確認する。"""
    paths = list(Path('scripts').glob('*e28*.py')) + [Path('scripts/render_e27_review.py')]
    long_functions = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        compile(tree, str(path), 'exec')
        long_functions.extend((str(path), n.name, n.end_lineno-n.lineno+1)
            for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.end_lineno-n.lineno+1 > 50)
    assert not long_functions, long_functions
    assert hashlib.sha256(Path('src/production_config.py').read_bytes()).hexdigest() == PRODUCTION_SHA
    conclusion = json.loads((OUT/'SCENE_CONCLUSION.json').read_text())
    assert conclusion['scene_observations'] == 480
    assert conclusion['scene_landing_evaluations'] == 89
    assert conclusion['multi_reasons'] == dict(uncertain_completion=86, board_unsettled=3)
    assert conclusion['multi_searches'] == 0 and conclusion['unavoidable_death_sec'] is None
    assert [v['matching'] for v in conclusion['single_sample_checks']] == [0,25,5,5,0]
    assert conclusion['first_display_p2_le_5'] == 2771.25
    with (OUT/'SCENE.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 480 and sum(r['reevaluated'] == 'True' for r in rows) == 89
    assert all(not r['p2_maximum_extra_counter_ojama'] and not r['p2_required_cancel_ojama'] for r in rows)
    equivalence = json.loads((OUT/'REPLAY_VERIFICATION.json').read_text())['equivalence']
    assert equivalence == dict(display='byte_identical', events='byte_identical', diagnostics='identical')
    verify_live()
    for path in (OUT/'SCENE_RAW.json', OUT/'live/SCENE_RAW.json'):
        if path.exists():
            path.with_suffix('.json.gz').write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    save_json(OUT/'VERIFICATION.json', dict(python_files=len(paths), function_limit=50,
        production_sha256=PRODUCTION_SHA, csv_rows=len(rows), equivalence=equivalence,
        scene_checks='passed', visual='VISUAL_VERIFICATION.json'))
    print('E28診断の構文・関数長・480行・原E27出力一致・本番設定不変: OK', flush=True)


if __name__ == '__main__':
    main()
