"""速さ修正版の独立コピーと配布。旧配布と本線の既定は保存する。"""

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text

from pathlib import Path
from functools import partial
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_live_obs_20261020b as build

build.BUILD = 'build_20261020d'
build.OUT = Path('D:/puyo_analyzer/packaging/build_20261020d_release')
build.FAST = Path('D:/puyo_analyzer/packaging/build_20261020d_stage_r3/PuyoLive')
build.BUNDLE = build.FAST
build.PRELOAD_ARCHIVES = True
build.PRECOMPILE_RUNTIME = Path(_public_dev_text('<work>/build_20261019a_stage/PuyoLive/python'))
build.write_bundle_zip = partial(build.write_bundle_zip, in_memory=True)
build.inventory = partial(build.inventory, preload=True)
build.OLD_ZIP = Path('D:/puyo_analyzer/packaging/build_20261020c_release/'
                     'PuyoLive_20261020c_USER_ONLY_DO_NOT_DISTRIBUTE.zip')
build.ENTRY = Path('D:/PuyoLive_OBS_20261020d_USER_ONLY')
# 手動試験も根本対策のCPU0〜11を使わない。小型機では許されたCPU集合を使う。
build.launchers.WRAPPER = build.launchers.WRAPPER.replace('import json', 'import json\nimport os').replace(
    'assert kernel.SetProcessAffinityMask(handle,255)',
    'allowed, system = ctypes.c_size_t(), ctypes.c_size_t()\n'
    '    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]\n'
    '    assert kernel.GetProcessAffinityMask(handle,ctypes.byref(allowed),ctypes.byref(system))\n'
    '    mask = allowed.value & (61440 if (os.cpu_count() or 1) >= 16 else allowed.value)\n'
    '    assert kernel.SetProcessAffinityMask(handle,mask)\n'
    "    os.environ['PUYO_OBS_CPU_ISOLATION'] = '0'\n"
    "    os.environ['PUYO_OBS_FAST_AUDIT'] = '1'\n"
    "    os.environ['PUYO_OBS_EXACT_SEED'] = '1'")
build.launchers.WRAPPER = build.launchers.WRAPPER.replace(
    "    run.mkdir(parents=True, exist_ok=False)",
    "    run.mkdir(parents=True, exist_ok=False)\n"
    "    temporary = run/'temp'\n"
    "    temporary.mkdir()\n"
    "    os.environ.update(TEMP=str(temporary), TMP=str(temporary))")
original_guide = build.launchers.guide


def guide(*args: object, **kwargs: object) -> str:
    return original_guide(*args, **kwargs)+'''
速さ修正版: 今回は起動元のCPU集合をメインとamaで共有し、すべてBelowNormalです。
4論理CPUを3+1へ分離した参考試行のama P95は3.86秒で、分離は採用していません。
分離機能は試験用に残し、8以上ではamaへ最大2論理CPUを割当て可能です。
PC1ではCPU12〜15だけを使用します。根本対策のCPU・GPU・優先度は変更しません。
amaのツモ種照合は、色違い・途中不一致の候補を先に除き、完全一致の条件と候補順を保ちます。
正式な速度測定は根本対策の重い計算が空くまで10分ごと・最大8時間待ち、
CPUクロック・同居負荷・空きメモリ6GiB以上を記録して別途行います。
PC1では本体・output・一時ファイルをD:に置きます。正常終了まで待って記録を保存してください。
C:の空き10GiB未満、または空きメモリ6GiB未満では新しい試験を開始しないでください。
速度門は未合格です。同居負荷のある4論理CPUの参考測定を、前版の8論理CPUと直接比較しません。
CPU共有の比較試行ではv4.9fのKeyErrorを確認しました。数値計算や欠番補完は変更していません。
表示欄にエラーが出た場合も記録を残し、正常終了を待ってPC1へ検収を依頼してください。
'''


build.launchers.guide = guide


def stage() -> None:
    """破損した途中コピーを使わず、完成ZIPからD:へ再構築する。"""
    build.extract_smoke_copy(build.OLD_ZIP, build.BUNDLE.parent, preload=True)
    build.refresh()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stage', 'refresh', 'seal'))
    parser.add_argument('--bundle', type=Path, default=build.BUNDLE)
    args = parser.parse_args()
    build.BUNDLE = args.bundle
    (stage if args.action == 'stage' else getattr(build, args.action))()
