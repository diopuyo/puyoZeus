"""実configure後の原cold loaderで旧衝突拒否と新入口成功を確認する。更新はしない。"""
from __future__ import annotations
from contextlib import ExitStack
import importlib.util
import sys
import proof_snapshot as P


def main() -> None:
    P.configure(P.OLD.configure)
    path = P.ROOT.parent / 'g2_directional_history_full_runtime_2026-09-09_v1/deps_full.py'
    spec = importlib.util.spec_from_file_location('_g2_snapshot_original_dependencies', path)
    deps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deps)
    assert not set(deps.FILES).intersection(sys.modules), 'reserved_alias_collision'
    sys.modules['connection'] = P.C
    try:
        with ExitStack() as stack:
            deps.load('connection', stack)
    except RuntimeError as error:
        assert str(error) == 'full_dependency:connection'
    else:
        raise AssertionError('旧衝突を拒否しなかった')
    finally:
        assert sys.modules.pop('connection') is P.C
    with ExitStack() as stack:
        loaded = deps.load('connection', stack)
        assert loaded is sys.modules['connection'] and loaded is not P.C
    assert 'connection' not in sys.modules
    print('original_cold_loader_old_collision_rejected_new_namespace_pass; actual_update_not_run')


if __name__ == '__main__':
    main()
