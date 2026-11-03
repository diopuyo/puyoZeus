"""静止監査の母数・重み・コード規約・資産ハッシュを独立照合する。"""
from __future__ import annotations
import argparse
import ast
import hashlib
import gzip
import json
from pathlib import Path
import subprocess
from scripts._audit_static_1003_extract import OUT, BASE, EVAL, SOURCES, guard, save_json
import numpy as np
import pandas as pd


def code_checks() -> list[dict]:
    """追加監査コードの構文、型注釈、50行上限を検査する。"""
    checks = []
    for path in sorted(Path('scripts').glob('_audit_static_1003_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            checks.append(dict(file=str(path), function=node.name, lines=node.end_lineno - node.lineno + 1,
                               returns=node.returns is not None,
                               arguments=all(a.annotation is not None for a in node.args.args)))
    print('規約違反', [c for c in checks if c['lines'] > 50 or not c['returns'] or not c['arguments']], flush=True)
    return checks


def provenance() -> None:
    """再現に必要な入力のバイト同一性を保持する。"""
    paths = [BASE / f'collect/records/{s}.jsonl.gz' for s in SOURCES]
    paths += [EVAL / 'labels.json', EVAL / 'set2/labels.json',
              Path('models/exchange_event_v1/G_fe.joblib'), Path('models/exchange_event_v1_r1_hgb/G_fe.joblib')]
    from scripts._audit_static_1003_tables import PHASE_ROOT, REFERENCE
    paths += [PHASE_ROOT / 'gfe_phase_stage1_20261002/predictions.csv',
              PHASE_ROOT / 'gfe_phase_reference_20261002/predictions.csv',
              REFERENCE / 'r2_baselines_148_20260926/design.npy',
              REFERENCE / 'r2_baselines_148_20260926/rows.csv']
    hashes = []
    for path in paths:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        hashes.append(dict(path=str(path.resolve()), sha256=digest, bytes=path.stat().st_size))
    save_json('provenance.json', dict(base_commit='802efbcbc38156e0bd521e9439a7090f2e56c549',
        current_commit=revision(), inputs=hashes))


def revision() -> str:
    """Windows所有worktreeのgitdirをWSLの読取パスに変換する。"""
    value = Path('.git').read_text().strip().removeprefix('gitdir: ').replace('\\', '/')
    directory = '/mnt/' + value[0].lower() + '/' + value[3:] if value[1:3] == ':/' else value
    return subprocess.check_output(['git', '--git-dir=' + directory, 'rev-parse', 'HEAD'], text=True).strip()


def numeric_checks() -> dict:
    """表示集計の足し戻しと成功入力・火力の対象漏れを検査する。"""
    f, g = (pd.read_pickle(OUT / name) for name in ('display_all.pkl', 'display_static.pkl'))
    assert f.game.nunique() == 115
    np.testing.assert_allclose(g.groupby('game').weight.sum(), 1 / g.game.nunique(), rtol=0, atol=1e-12)
    strata = pd.read_csv(OUT / 'display_strata.csv')
    all_row = strata[strata.axis == 'all'].iloc[0]
    for name in ('U_slow', 'U_legacy_max', 'amount_diff', 'chain_diff', 'ojama_diff'):
        rows = strata[strata.axis == name]
        assert rows.rows.sum() == len(g)
        np.testing.assert_allclose(rows.weight.sum(), 1, atol=1e-12)
        np.testing.assert_allclose(np.dot(rows.weight, rows.R1s_LL), all_row.R1s_LL, atol=1e-12)
    fire = pd.read_pickle(OUT / 'fire_joined.pkl')
    expected = g[g.stable_zero]
    excluded = json.loads((OUT / 'fire_exclusions.json').read_text())
    assert all(r['reason'] == 'confirmed_board_missing' for r in excluded)
    assert len(fire) + len(excluded) == len(expected), (len(fire), len(expected))
    assert not fire[['source_record', 't']].duplicated().any()
    return dict(games=115, gfe_rows=len(g), fire_rows=len(fire),
                per_game_equal_weight=True, strata_reconstruction=True,
                available_confirmed_fire_complete=True, confirmed_board_missing=len(excluded))


def fire_checks() -> dict:
    """監査で使った高速探索と既存既定探索を保存盤面で比較する。"""
    from src.board import Board
    from src import indicators_v2 as iv
    from scripts._audit_static_1003_fire import pair
    count = 0
    for source in ('c1', 's1', 's6'):
        with gzip.open(OUT / f'{source}_samples.json.gz', 'rt') as stream:
            samples = json.load(stream)
        for sample in samples[::max(1, len(samples) // 3)][:3]:
            for side in sample['sides']:
                board = Board.from_list(side['confirmed_board']['board']['array'])
                args = (board, pair(side['next_pair']), pair(side['dnext_pair']), 0)
                slow, fast = iv.near_future_fire_power(*args), iv.near_future_fire_power(*args, fast_expand=True)
                assert slow == fast
                count += 1
    return dict(boards=count, default_fast_identity=True)


def main() -> None:
    """完了前はコードだけ、完了後は全出力を検査する。"""
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-only', action='store_true')
    args = parser.parse_args()
    checks = code_checks()
    if args.code_only:
        return
    assert all(c['lines'] <= 50 and c['returns'] and c['arguments'] for c in checks)
    provenance()
    save_json('VERIFICATION.json', dict(code=checks, numeric=numeric_checks(), fire=fire_checks()))
    print('独立検証完了', flush=True)


if __name__ == '__main__':
    main()
