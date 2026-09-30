"""配布する CNN (.pt) を ONNX へ書き出す。ONNX バックエンド (src/cnn_onnx.py、既定 OFF) 用。

    <PuyoLive/python/python.exe> packaging/export_onnx.py <models ディレクトリ> [.pt 名 ...]

出力: <models>/onnx/<重みの内容ハッシュ>.onnx と index.json (由来台帳)。書き出しごとに
乱数入力で torch と ONNX Runtime の logits を比べ、最大絶対差と argmax 一致数を台帳へ記録する
(本番の合否判定ではなく書き出しの健全性確認。合否は packaging/verify_onnx_parity.py)。
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_MODELS = ('cnn_phase_b_large_v2.pt', 'cnn_global_best.pt', 'cnn_best.pt')
SANITY_BATCH = 256
SANITY_SEED = 0


def sanity(state: dict, path: Path) -> dict:
    """乱数入力での torch 対 ONNX の差 (書き出しの取り違え検出用)。"""
    import torch
    from src import cnn_onnx
    from src.patch_classifier import CnnPatchClassifier, CnnPatchClassifierLarge, PATCH_RESIZE_H, PATCH_RESIZE_W
    large = state[next(iter(state))].shape[0] == cnn_onnx.LARGE_FIRST_CONV_CHANNELS
    model = (CnnPatchClassifierLarge if large else CnnPatchClassifier)()
    model._model.load_state_dict(state)
    model._model.eval()
    generator = torch.Generator().manual_seed(SANITY_SEED)
    batch = torch.rand(SANITY_BATCH, 6, PATCH_RESIZE_H, PATCH_RESIZE_W, generator=generator)
    with torch.no_grad():
        expected = model._model(batch).numpy()
    digest = path.stem
    actual = cnn_onnx.run_logits(cnn_onnx.session_for(digest, path.parent), batch.numpy())
    return dict(samples=SANITY_BATCH, max_abs_logit_diff=float(np.abs(expected - actual).max()),
                argmax_equal=int((expected.argmax(1) == actual.argmax(1)).sum()))


def main() -> None:
    import onnxruntime
    import torch
    from src import cnn_onnx
    models = Path(sys.argv[1])
    names = sys.argv[2:] or list(DEFAULT_MODELS)
    out = models / 'onnx'
    entries: dict[str, dict] = {}
    for name in names:
        state = torch.load(str(models / name), map_location='cpu', weights_only=True)
        digest, path = cnn_onnx.export_state(state, out)
        entries[digest] = dict(source=name, torch=torch.__version__, onnxruntime=onnxruntime.__version__,
                               opset=cnn_onnx.OPSET, sanity=sanity(state, path))
        print(name, digest, entries[digest]['sanity'])
    cnn_onnx.write_index(out, entries)


if __name__ == '__main__':
    main()
