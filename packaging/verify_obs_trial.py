"""別出力先に組み立てたOBS試験版のソース・モデル・MANIFESTを照合する。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = '6113e42b8d3aadad6251f0749ad09281d7c63a65fc09a4cc659e68694ee2903c'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('output', type=Path)
    options = parser.parse_args()
    app = options.bundle / 'app'
    sys.path.insert(0, str(app))
    from src.phase_j.launcher_manifest import verify_manifest
    report = verify_manifest(app, app / 'MANIFEST.json', options.bundle / 'python/Lib/site-packages')
    assert report.ok, report.summary()
    paths = [path for path in app.rglob('*.py') if '__pycache__' not in path.parts]
    different = [str(path.relative_to(app)) for path in paths
                 if path.read_bytes() != (ROOT / path.relative_to(app)).read_bytes()]
    assert paths and not different, different
    model_hash = sha256((app / 'models/b3/model.joblib').read_bytes()).hexdigest()
    assert model_hash == MODEL_SHA256, model_hash
    result = dict(manifest=asdict(report), source_files_identical=len(paths),
                  model_sha256=model_hash)
    options.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
