"""21bの資産と通常計算で初回importを計装する開発用診断。正式計測には使わない。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_live_obs_20261020 as smoke

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('D:/puyo_analyzer/packaging/build_20261021c_work')


def main() -> None:
    names = ['live_process', 'live_bridge', 'live_eval_worker', 'live_stage_timing',
             'live_obs_prepare', 'live_obs_recognition_timing']
    plan = WORK/'diagnostic_source_plan.json'
    plan.write_text(json.dumps(dict(changed=[dict(name='src/phase_j/'+n+'.py') for n in names])), encoding='utf-8')
    sys.argv = [__file__, '--bundle', 'D:/puyo_analyzer/packaging/build_20261021b_stage/PuyoLive',
        '--out', str(WORK/'first_import_diagnostic_fixed'), '--video', 'C:/puyo1_tools/soak/media/video_d04.mp4',
        '--start-sec', '600', '--duration', '180', '--startup-timeout', '1200', '--port', '18857',
        '--ffmpeg-dll-dir', _public_dev_text('<work>/media_tmp/b2_cv_20261020/cv2'),
        '--fast-audit', '--exact-seed', '--stdlib-cache', '--deep-timing',
        '--source-overlay', str(ROOT), '--source-plan', str(plan)]
    original = smoke.launch_command
    def command(args: object) -> list[str]:
        result = original(args)
        result[0] = _public_dev_text('<work>/build_20261019a_stage/PuyoLive/python/python.exe')
        return result
    smoke.launch_command = command
    smoke.run_cli()


if __name__ == '__main__':
    main()
