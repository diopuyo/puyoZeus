"""フラグが実際に効いたかの確認 (測定器の健全性): ONNXスレッド数・色補正buffer再利用。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[1])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

import json, os, sys
sys.path.insert(0, _public_dev_text('<work>\\wt_live_integ')); sys.path.insert(1, _public_dev_text('<work>\\wt_live_integ\\scripts'))
os.chdir(_public_dev_text('<work>\\assets\\baseline_b\\app'))
os.environ['PUYO_CNN_BACKEND'] = 'onnx'
from pathlib import Path
index = json.loads(Path('models/onnx/index.json').read_text(encoding='utf-8'))
digest = next(iter(index.get('models', index)))
for flag in ('0', '1'):
    os.environ['PUYO_LIVE_ONNX_PARALLEL'] = flag
    from src import cnn_onnx
    s = cnn_onnx.session_for(digest, Path('models/onnx'))
    o = s.get_session_options()
    try:
        spin = o.get_session_config_entry('session.intra_op.allow_spinning')
    except RuntimeError:
        spin = 'unset(既定=スピンあり)'
    print('onnx_parallel', flag, 'intra', o.intra_op_num_threads, 'inter', o.inter_op_num_threads, 'allow_spinning', spin)
os.environ['PUYO_LIVE_COLOR_BUFFER'] = '1'
from src.phase_j.live_color_buffer import configured_corrector
import live_integ_gate as g
c = configured_corrector(); buf = c.__self__
for n, f in enumerate(g.DecodedFrames('C:/puyo1_tools/soak/media/video_d04.mp4', 650.0, 30)):
    out = c(f); del out
print('color_buffer reused', buf.reused, 'allocated', buf.allocated)
