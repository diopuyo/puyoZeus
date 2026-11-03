"""B3 アンサンブルの 1 行の推論時間を、配信と同じ条件で測る。

配信 (b3_async._compute) と同じく threadpool_limits(1) の中で、1 行ずつ predict_row を呼ぶ。測る単位は
「predict_row 1 回 (行の列取得 + 全メンバーの predict_proba + 平均)」で、1 コマや 1 更新ではない。
入力の行は dl1k の M3 学習行 (+正味の列を台帳の導出関数で足したもの) からの無作為抽出 (実データの値)。

使い方 (WSL の venv で):
  python -m scripts.b3_ensemble_bench --model large_seed1003.joblib --copies 3
      同じモデルを 3 回読み込んで 3 本並べる (最終学習のモデルが届く前の代用。1 本の出力とのビット一致も確かめる)
  python -m scripts.b3_ensemble_bench --model a.joblib --model b.joblib --model c.joblib --aggregate mean_p
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

# dl1k の M3 学習行 (469 列・実データ)。WSL からは C: が /mnt/c に見える (D: は見えない)。
ROWS_PARQUET = Path(_public_dev_text('<work>/b3_c2_wiring_20261025/m3_rows_video_c106.parquet') if os.name == 'nt'
                    else _public_dev_text('<work>/b3_c2_wiring_20261025/m3_rows_video_c106.parquet'))
DEFAULT_ROWS, DEFAULT_WARMUP, DEFAULT_SEED = 400, 20, 0
MS = 1000.0
PERCENTILES = (50, 95)


def real_rows(columns: Sequence[str], n_rows: int, seed: int = DEFAULT_SEED,
              parquet: Path = ROWS_PARQUET) -> list[dict[str, float]]:
    """学習行 + 正味の列から n_rows 行を無作為に取り、配信の 1 行 (dict) と同じ形にする。欠測は NaN。"""
    import pandas as pd
    from src.b3_net_ledger import derive_frame
    frame = pd.read_parquet(parquet)
    full = pd.concat([frame, derive_frame(frame)], axis=1)
    needed = list(dict.fromkeys([*columns, 'mode_exchange']))
    missing = [c for c in needed if c not in full.columns]
    if missing:
        raise KeyError(f'学習行にモデルの列が無い: {missing[:8]} (計{len(missing)})')
    picked = full.sample(n=min(n_rows, len(full)), random_state=seed)
    values = {c: picked[c].to_numpy(dtype=np.float64, na_value=np.nan) for c in needed}
    return [{c: float(values[c][i]) for c in needed} for i in range(len(picked))]


def time_rows(model: Any, rows: Sequence[dict], warmup: int = DEFAULT_WARMUP) -> tuple[list[float], list[float]]:
    """(1 行ごとの ms, 出力の確率)。先頭 warmup 行は初回の準備 (列取得・スレッドプール探索) を含むので ms から外す。"""
    from threadpoolctl import threadpool_limits
    costs, outputs = [], []
    with threadpool_limits(limits=1):   # b3_async._compute と同じ外側の制限
        for index, row in enumerate(rows):
            started = perf_counter()
            probability = model.predict_row(row)
            elapsed = (perf_counter() - started) * MS
            outputs.append(probability)
            if index >= warmup:
                costs.append(elapsed)
    return costs, outputs


def summarize(costs: Sequence[float]) -> dict[str, float]:
    values = np.asarray(costs, dtype=np.float64)
    result = {f'p{q}_ms': float(np.percentile(values, q)) for q in PERCENTILES}
    return dict(n=int(len(values)), mean_ms=float(values.mean()), max_ms=float(values.max()), **result)


def same_bits(a: Sequence[float], b: Sequence[float]) -> bool:
    return bool(np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), equal_nan=True))


def build_members(paths: Sequence[Path], copies: int) -> list[Any]:
    """モデルを読み込む。copies > 1 なら先頭のモデルを copies 回、別の object として読み込んで並べる。"""
    from src.b3_train_model import load_model
    if copies > 1:
        if len(paths) != 1:
            raise SystemExit('--copies は --model が 1 つのときだけ使えます')
        paths = [paths[0]] * copies
    return [load_model(path) for path in paths]


def measure(members: Sequence[Any], aggregate: str, n_rows: int, warmup: int) -> dict[str, Any]:
    """1 本 (先頭のモデル) と全メンバーの平均の、行ごとの推論時間と出力の一致を測る。"""
    from src.b3_ensemble import EnsembleTwoStateModel
    ensemble = EnsembleTwoStateModel(members, aggregate)
    columns = sorted(set(members[0].static.columns) | set(members[0].exchange.columns))
    rows = real_rows(columns, n_rows)
    single_ms, single_out = time_rows(members[0], rows, warmup)
    ensemble_ms, ensemble_out = time_rows(ensemble, rows, warmup)
    modes = np.array([row['mode_exchange'] for row in rows])
    return dict(rows=len(rows), warmup_rows=warmup, rows_static=int((modes == 0).sum()),
                rows_exchange=int((modes == 1).sum()), members=len(members), aggregate=aggregate,
                unit='predict_row 1 回 (threadpool_limits(1)・1 行ずつ)',
                single=summarize(single_ms), ensemble=summarize(ensemble_ms),
                ratio_p50=float(np.percentile(ensemble_ms, 50) / np.percentile(single_ms, 50)),
                bit_identical_to_single=same_bits(single_out, ensemble_out))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    from src.b3_ensemble import AGGREGATES, DEFAULT_AGGREGATE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, action='append', required=True, help='TwoStateModel の joblib (繰り返し可)')
    parser.add_argument('--copies', type=int, default=1, help='先頭のモデルをこの本数だけ読み込んで並べる (代用)')
    parser.add_argument('--aggregate', choices=AGGREGATES, default=DEFAULT_AGGREGATE)
    parser.add_argument('--rows', type=int, default=DEFAULT_ROWS)
    parser.add_argument('--warmup', type=int, default=DEFAULT_WARMUP)
    parser.add_argument('--out', type=Path, help='結果の JSON の保存先')
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(sys.argv[1:] if argv is None else argv)
    result = measure(build_members(options.model, options.copies), options.aggregate, options.rows, options.warmup)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if options.out is not None:
        options.out.write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
