"""組み立てた配布物の app で、段1 の同値試験 (scripts/verify_live_x_direct_parity.py の方式) を流す。

21c 保存入力 5,921 行を v4.9f 欄へ OFF と ON で流し、OFF が元の走行を全行再現し、OFF と ON が
計時以外で全行一致することを確かめる。元の台は src を worktree から読むが、ここでは製品コード src を
配布 app から読み、測定台 (scripts) だけを worktree から読む。spawn の子も同じ sys.path を継ぐ。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

WORKTREE = Path(__file__).resolve().parents[1]
BUNDLE_APP = Path('D:/puyo_analyzer/packaging/build_20261023_obs_x/stage/PuyoLive/app')
PARITY = WORKTREE/'scripts/verify_live_x_direct_parity.py'
PRODUCT_MODULES = ('src.phase_j.live_v49f', 'src.phase_j.live_v49f_async', 'src.phase_j.live_x_direct',
                   'src.phase_j.live_v49f_origins', 'src.prefire_origin_stream')


def isolate_paths() -> None:
    """配布 app を先頭に置き、実行系の旧 app を外し、worktree は測定台の読み込み用に末尾だけに置く。"""
    old_app = (Path(sys.executable).resolve().parent.parent/'app').resolve()
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() not in (old_app, WORKTREE.resolve())]
    sys.path.insert(0, str(BUNDLE_APP))
    sys.path.append(str(WORKTREE))
    # 測定台は import 時に worktree を sys.path 先頭へ足す。先に読み込み、足された分を外して順序を戻す。
    import scripts.measure_live_x_effects  # noqa: F401
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != WORKTREE.resolve()]+[str(WORKTREE)]


def product_origins() -> dict[str, str]:
    import importlib
    origins = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in PRODUCT_MODULES}
    # D: はネットワークドライブで resolve() が UNC になるため、両辺を resolve して比べる。
    outside = {k: v for k, v in origins.items() if not Path(v).is_relative_to(BUNDLE_APP.resolve())}
    assert not outside, f'配布 app 以外から製品コードを読んだ: {outside}'
    return origins


def guarded(run_arm):
    """各腕の開始時に、配布 app が worktree より先にあることを確かめる (spawn の子が継ぐ順序)。"""
    def wrapper(*args, **kwargs):
        resolved = [Path(p).resolve() for p in sys.path if p]
        assert resolved.index(BUNDLE_APP.resolve()) < resolved.index(WORKTREE.resolve()), sys.path
        return run_arm(*args, **kwargs)
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.chdir(BUNDLE_APP)
    isolate_paths()
    spec = importlib.util.spec_from_file_location('verify_live_x_direct_parity', PARITY)
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    parity.ROOT = BUNDLE_APP      # main() が sys.path 先頭へ足す場所を配布 app にする
    parity.run_arm = guarded(parity.run_arm)
    origins = product_origins()
    sys.argv = [str(PARITY), '--out', str(args.out)]
    parity.main()
    receipt_path = args.out/'receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    receipt.update(product_module_files=origins, bundle_app=str(BUNDLE_APP), parity_script=str(PARITY))
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=receipt['passed'], product_module_files=origins), ensure_ascii=False), flush=True)
    if not receipt['passed']:
        raise SystemExit('同値試験 不合格')


if __name__ == '__main__':
    main()
