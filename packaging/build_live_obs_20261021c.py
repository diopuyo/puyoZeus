"""21bを保存し、同値確認した認識高速化・起動前準備を21cへ同梱する。"""
from pathlib import Path
import argparse
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'packaging')]
import build_live_obs_20261021b as previous

build = previous.build
build.BUILD = 'build_20261021c'
build.OUT = Path('D:/puyo_analyzer/packaging/build_20261021c_release')
build.BUNDLE = Path('D:/puyo_analyzer/packaging/build_20261021c_stage/PuyoLive')
build.FAST = build.BUNDLE
build.OLD_ZIP = Path('D:/puyo_analyzer/packaging/build_20261021b_release/'
    'PuyoLive_20261021b_USER_ONLY_DO_NOT_DISTRIBUTE.zip')
build.ENTRY = Path('D:/PuyoLive_OBS_20261021c_USER_ONLY')
build.launchers.WRAPPER = build.launchers.WRAPPER.replace(
    "    os.environ['PUYO_OBS_EXACT_SEED'] = '1'",
    "    os.environ['PUYO_OBS_EXACT_SEED'] = '1'\n"
    "    os.environ['PUYO_OBS_PREPARE_EXACT'] = '1'\n"
    "    os.environ['PUYO_OBS_DEEP_TIMING'] = '1'")
guide_before = build.launchers.guide
populate_before = build.launchers.populate


def populate(folder: Path, bundle: Path, external: bool) -> None:
    populate_before(folder, bundle, external)
    shutil.copy2(ROOT/'packaging/summarize_live_obs_actual_latency.py', folder/'summarize_live_obs_actual_latency.py')


def guide(*args: object, **kwargs: object) -> str:
    bundle = args[0] if args else Path('.')
    external = args[1] if len(args) > 1 else kwargs.get('external', False)
    python = str(Path(bundle)/'python/python.exe') if external else 'python/python.exe'
    return guide_before(*args, **kwargs)+f'''
21c: 置き完了照合・発火前画像の固定HSV分類を同値のネイティブ実装へ変更しました。
初回通知に必要なimportは起動準備で済ませます。観測の間引きや更新間隔の変更はありません。
固定モデルのパス解決とONNXセッション生成も起動時に済ませます。
21bの起動フォルダは控えとしてそのまま残しています。
取得からHTTP公開までの実遅延（先頭20分）も同じ記録から集計できます:
& "{python}" summarize_live_obs_actual_latency.py output/日時
output/日時/actual_latency.json にP50・P95・最大・500ms超件数を保存します。
表示したB3の元入力からの遅れと、最新認識からの遅れを分けて記録します。
'''


build.launchers.guide = guide
build.launchers.populate = populate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('extract', 'refresh', 'seal'))
    action = parser.parse_args().action
    if action == 'extract':
        from bundle_archive import extract_smoke_copy
        extract_smoke_copy(build.OLD_ZIP, build.BUNDLE.parent, preload=True)
    elif action == 'refresh':
        previous.refresh()
    else:
        build.seal()
