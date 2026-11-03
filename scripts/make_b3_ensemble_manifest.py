"""B3 アンサンブルのマニフェスト (JSON) を作る。モデルの SHA256 を計算して並べ、全モデルの列が同じことも確かめる。

使い方 (WSL の venv で。モデルはマニフェストと同じフォルダに置く):
  python -m scripts.make_b3_ensemble_manifest --out <folder>/ensemble.json <folder>/a.joblib <folder>/b.joblib <folder>/c.joblib
出力されたマニフェストの SHA256 を src/phase_j/live_obs_trial.py の C2_ENSEMBLE_SHA256 へ登録する (登録は別の判断)。
既存のファイルは上書きしない。--aggregate は mean_logit (既定) か mean_p。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.b3_ensemble import AGGREGATES, DEFAULT_AGGREGATE, build_manifest, load_ensemble_model, sha256_of_file


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', type=Path, nargs='+', help='TwoStateModel の joblib (マニフェストと同じフォルダ)')
    parser.add_argument('--out', type=Path, required=True, help='書き出すマニフェスト (既存なら止まる)')
    parser.add_argument('--aggregate', choices=AGGREGATES, default=DEFAULT_AGGREGATE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = build_manifest(options.models, options.aggregate)
    if options.out.parent.resolve() != options.models[0].parent.resolve():
        raise SystemExit('マニフェストはモデルと同じフォルダへ書いてください')
    with options.out.open('x', encoding='utf-8') as stream:      # 既存ファイルは上書きしない
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    ensemble = load_ensemble_model(options.out)                   # 読み込めること・全モデルの特徴列が同じこと・SHA256 の照合
    print(json.dumps(dict(manifest=str(options.out), manifest_sha256=sha256_of_file(options.out), aggregate=ensemble.aggregate,
                          members=len(ensemble.members), static_columns=len(ensemble.static.columns),
                          exchange_columns=len(ensemble.exchange.columns)), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
