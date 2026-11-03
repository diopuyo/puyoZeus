"""配布Pythonでjoblibを読み、同一入力の一行推論を照合する。組立はしない。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np

from src.b3_source import sha256_file
from src.b3_train_model import load_model

FIXTURE_ROWS = 256


def decision_scores(model: object, rows: list[dict]) -> np.ndarray:
    from threadpoolctl import threadpool_limits
    scores = []
    with threadpool_limits(limits=1):
        for row in rows:
            ev = model.exchange if row['mode_exchange'] else model.static
            values = np.asarray([[row[c] for c in ev.columns]], dtype=np.float32)
            scores.append(float(ev.model.decision_function(values)[0]))
    return np.asarray(scores)


def make_fixture(rows_path: Path, model: object) -> dict:
    import pandas as pd
    rows = pd.read_parquet(rows_path)
    rows = rows[(rows.orientation == 0) & (rows.frame_idx <= 36000)]
    chosen = np.linspace(0, len(rows)-1, min(FIXTURE_ROWS, len(rows)), dtype=int)
    rows = rows.iloc[chosen]
    expected = model.predict(rows)
    columns = sorted({c for ev in (model.static, model.exchange) for c in ev.columns} | {'mode_exchange'})
    records = rows[columns].to_dict('records')
    return dict(rows=records, expected=expected.tolist(), scores=decision_scores(model, records).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--rows', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    if args.rows:
        args.fixture.write_text(json.dumps(make_fixture(args.rows, model)), encoding='utf-8')
    fixture = json.loads(args.fixture.read_text(encoding='utf-8'))
    expected = np.asarray(fixture['expected'])
    actual = np.asarray([model.predict_row(row) for row in fixture['rows']])
    scores = decision_scores(model, fixture['rows'])
    import sklearn, joblib
    result = dict(rows=len(actual), modes=sorted({int(r['mode_exchange']) for r in fixture['rows']}),
        bit_equal=bool(np.array_equal(expected.view(np.uint64), actual.view(np.uint64))),
        max_abs=float(np.max(np.abs(actual-expected))), model_sha256=sha256_file(args.model),
        decision_max_abs=float(np.max(np.abs(scores-np.asarray(fixture['scores'])))),
        decision_bit_equal=bool(np.array_equal(scores, fixture['scores'])),
        python=sys.executable, platform=platform.platform(), sklearn=sklearn.__version__, joblib=joblib.__version__)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
