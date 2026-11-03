"""m配布と同じ起動スモークを内部試験版で行い、D参照を遮断する。"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(_public_dev_text('<work>/coalesce2/smoke_m.py'))


def main() -> None:
    """既存スモークは読取だけ。出力先と試験機器番号だけを変更する。"""
    mode = sys.argv[1]
    assert mode in ('off', 'on')
    source = SOURCE.read_text(encoding='utf-8')
    before = "OUT=Path('C:/puyo1_work/coalesce2')/('smoke_missing' if missing else 'smoke_present')"
    assert source.count(before) == 1
    tag = sys.argv[2] if len(sys.argv) > 2 else 'smoke'
    source = source.replace(before, f'OUT=Path({str(ROOT / "logs/live_parallel" / (tag+"_"+mode))!r})')
    assert source.count('assert user.device_index == 3') == 1
    source = source.replace('assert user.device_index == 3', 'assert user.device_index == 2')
    namespace = dict(__name__='__main__', __file__=str(SOURCE))
    exec(compile(source, str(SOURCE), 'exec'), namespace)
    if mode == 'on':
        from src import cnn_onnx
        assert namespace['user'].live_onnx_parallel
        assert namespace['user'].live_color_buffer
        assert cnn_onnx._SESSIONS
        for session in cnn_onnx._SESSIONS.values():
            options = session.get_session_options()
            assert options.intra_op_num_threads == 2
            assert options.get_session_config_entry('session.intra_op.allow_spinning') == '0'


if __name__ == '__main__':
    main()
