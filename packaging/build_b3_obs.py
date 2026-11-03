"""採用済みv6モデルと検証済みwheelでOBS配布物を組み立てる。"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'packaging'))
WORK = Path('D:/puyo_analyzer/packaging')
ARTIFACTS = Path('D:/puyo_analyzer/b3_data/codex_tasks/b3_realtime_v6_obs')
WHEEL = ARTIFACTS.parent / 'b3_realtime_v5swap/wheels/puyo_core-0.1.0-cp312-cp312-win_amd64.whl'
WHEEL_SHA256 = '7f631aadd7478b16a86d6c84fb7b1a3e44971c8b77be95bcb720959a66693bee'
TABLE = Path('D:/puyo_analyzer/b3_data/timing_v1/timing_table.json')
BOUND_COLUMNS = {'ta_bound_p1', 'ta_bound_p2'}


def validate(model_path: Path) -> None:
    """v4の誤投入とwheel違いを、配布物に触れる前に止める。"""
    from src.b3_train_model import load_model
    if sha256(WHEEL.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise ValueError('puyo_core wheelのSHA256が検証済み版と違います')
    model = load_model(model_path)
    if not all(e is not None and BOUND_COLUMNS <= set(e.columns)
               for e in (model.static, model.exchange)):
        raise ValueError('v5の末尾2列がないモデルです')


def main() -> None:
    """既存OBS入力設定を保存・復元し、B3モデルを差し替えて組み立てる。"""
    from build_bundle import build, DEFAULT_ASSET_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    args = parser.parse_args()
    validate(args.model)
    config = WORK / 'build/PuyoLive/puyo_live.json'
    saved = config.read_bytes() if config.is_file() else None
    shutil.copy2(WHEEL, WORK / 'native_build/wheels' / WHEEL.name)
    try:
        result = build(WORK, DEFAULT_ASSET_ROOT, False, reuse_python=True,
                       with_onnx=True, onnx_cache=WORK / 'onnx_cache',
                       b3_model=args.model, b3_timing_table=TABLE)
    finally:
        if saved is not None:
            config.write_bytes(saved)
    result['b3_model_sha256'] = sha256(args.model.read_bytes()).hexdigest()
    result['b3_timing_sha256'] = sha256(TABLE.read_bytes()).hexdigest()
    result['puyo_core_wheel_sha256'] = WHEEL_SHA256
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / 'bundle.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
