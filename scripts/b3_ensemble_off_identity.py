"""b3_ensemble が OFF のとき、変更前のコードと出力が同じことを確かめる (--root に変更前 / 変更後の src を指す)。

比べるもの (どちらの root でも同じ入力で作り、JSON にして SHA256 を並べる):
  - launcher: parse_user_config → build_pipeline_config の出力 (puyo_live.json → pipeline_config.json の変換) 数種
  - LiveB3 の b3_config.json のバイト列 (旧 B3 / M3 / C2 の 3 構成。モデルは列の名前だけを持つ仮のもの)
  - 子 process の環境へ届く PUYO_B3_* の鍵の集合 (live_obs_trial.configure の OFF と C2)
使い方 (変更前の src は `git archive <旧コミット> src config | tar -x -C <空フォルダ>` で取り出す):
  PYTHONPATH=<root> python scripts/b3_ensemble_off_identity.py --root <root> --out result.json
  変更前と変更後の result.json を比べる。launcher と b3_config は全て同じ SHA256、configure_env は
  変更後だけに PUYO_B3_ENSEMBLE=0 の 1 鍵が増える (OFF を子 process へ明示する鍵で、挙動は変えない)のが期待。
  作業フォルダ (一時フォルダ) の絶対パスは、比べる前に <work> へ置き換える。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PIPELINE_BASE = dict(source='dshow', device_name='OBS Virtual Camera', b3_judgment=True, b3_async=True,
                     b3_model='model.joblib', b3_timing_table='timing.json')
M3_FLAGS = dict(live_v49f=True, b3_m3=True, b3_time_gate=True, b3_ledger_mode=True, b3_ledger_landing=True,
                live_score_cadence=True, live_static_exact=True, live_ama_plain=True)
SCENARIOS: dict[str, dict[str, Any]] = {
    'plain': dict(device_name='OBS Virtual Camera'),
    'b3_only': dict(PIPELINE_BASE),
    'm3': dict(PIPELINE_BASE, **M3_FLAGS),
    'c2': dict(PIPELINE_BASE, **M3_FLAGS, b3_c2=True, b3_skip_icon_reading=True),
}
HEAD_COLUMNS = 6                      # 仮のモデルが持つ列の数 (列仕様の先頭から)


WORK_MARK = '<work>'      # 作業フォルダ (実行ごとに違う一時フォルダ) の絶対パスを、比べる前にこの印へ置き換える


def digest(text: str, folder: Path) -> str:
    """作業フォルダのパスだけを印に置き換えた文字列の SHA256 (実行ごとに違う一時フォルダのパスを比較から外す)。"""
    return hashlib.sha256(text.replace(str(folder), WORK_MARK).encode('utf-8')).hexdigest()


def launcher_outputs(folder: Path) -> dict[str, str]:
    from src.phase_j.launcher import build_pipeline_config, parse_user_config
    (folder/'model.joblib').write_bytes(b'fake')
    (folder/'timing.json').write_text('{}', encoding='utf-8')
    return {name: digest(json.dumps(build_pipeline_config(parse_user_config(dict(data), folder)), ensure_ascii=False), folder)
            for name, data in SCENARIOS.items()}


def fake_model(columns: list[str]) -> SimpleNamespace:
    evaluator = SimpleNamespace(columns=list(columns))
    return SimpleNamespace(static=evaluator, exchange=evaluator)


def b3_config_bytes(folder: Path, name: str, m3: bool, c2: bool) -> str:
    """LiveB3 を仮のモデルで作り、b3_config.json の SHA256 を返す (資産の読込と探索の本体は使わない)。"""
    from src import puyo_core_bridge as native
    from src.b3_column_spec import NET_LEDGER_FEATURE_NAMES, build_column_specs
    from src.b3_timing import TimingTable
    from src.phase_j import live_b3, live_obs_trial
    os.environ.update({'PUYO_B3_M3': str(int(m3)), live_obs_trial.C2_ENV: str(int(c2)),
                       live_obs_trial.NET_LEDGER_ENV: str(int(c2)), live_obs_trial.SKIP_ICON_ENV: str(int(c2))})
    names = [s.name for s in build_column_specs(m3) if s.role == 'feature' and 'icon' not in s.name][:HEAD_COLUMNS]
    columns = names + (list(NET_LEDGER_FEATURE_NAMES)[:2] if c2 else [])
    native.NATIVE_AVAILABLE, native._native = True, SimpleNamespace(b3_search_py=None)
    live_b3.load_model = lambda path: fake_model(columns)
    work = folder/name
    work.mkdir()
    (work/'timing.json').write_text(TimingTable().to_json(), encoding='utf-8')
    (work/'model.joblib').write_bytes(b'fake-model')
    out = work/'out'
    out.mkdir()
    live = live_b3.LiveB3(work/'model.joblib', work/'timing.json', out, trial_helpers=False)
    live.close()
    return digest((out/'b3_config.json').read_text(encoding='utf-8'), folder)


def configure_env_keys(folder: Path) -> dict[str, list[str]]:
    """live_obs_trial.configure の OFF が子 process の環境へ書く PUYO_B3_* の鍵 (鍵の集合だけ。値は別に見る)。"""
    from hashlib import sha256
    from src.phase_j import live_obs_trial as trial
    model = folder/'model.joblib'
    result = {}
    for name, c2 in (('off_m3', False), ('c2', True)):
        for key in [k for k in os.environ if k.startswith('PUYO_B3_')]:
            del os.environ[key]
        trial.M3_SHA256 = trial.C2_SHA256 = sha256(model.read_bytes()).hexdigest()
        options = SimpleNamespace(b3_judgment=True, b3_async=True, b3_m3=True, b3_time_gate=True, live_ama_plain=True,
                                  live_score_cadence=True, live_static_exact=True, b3_c2=c2, b3_skip_icon_reading=c2,
                                  b3_model=model)
        trial.configure(options, argparse.ArgumentParser())
        result[name] = sorted(f'{k}={v}' for k, v in os.environ.items() if k.startswith('PUYO_B3_'))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    options = parser.parse_args()
    sys.path.insert(0, str(options.root.resolve()))
    os.chdir(options.root)
    with tempfile.TemporaryDirectory() as raw:
        folder = Path(raw)
        result = dict(root=str(options.root), launcher=launcher_outputs(folder))
        result['b3_config'] = {name: b3_config_bytes(folder, name, m3, c2)
                               for name, m3, c2 in (('legacy', False, False), ('m3', True, False), ('c2', True, True))}
        result['configure_env'] = configure_env_keys(folder)
    options.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(options.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
