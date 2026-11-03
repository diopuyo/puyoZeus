"""run_live_x_replay.py 専用の子の入口。直接起動しない (起動口は run_live_x_replay.py だけ)。

Windows の spawn 子もこのファイルを読み直すので、環境変数があれば全 process に同じ差し替えを入れる。
ディスク上の配布物は変えない (工程共通 §3: 実行時に差し替え、件数を受領票へ残す)。
- 配布 app を基準に、この worktree の変更ソースだけを読み込む (packaging/run_live_obs_overlay.py と同じ方式)。
- 録画の入力: 左上 1636x920 → 1920x1080 INTER_AREA、601→709、記録秒 = 録画秒 + OFFSET (OBS_VS_VIDEO §6)。
  plan の crop が None なら切り抜かない (video_30 のようにゲーム画面そのものの入力。拡大は VideoFileSource 自身が行う)。
- 認識設定: 21c (dshow) と同じ enable_placement_signal_reconcile を使う (動画入力では既定が変わるため)。
- F 条件だけ realtime=False。L 条件は配布と同じ realtime=True (コマ落ち・壁時計門あり)。
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

ENV_PLAN = 'LIVE_X_REPLAY_CHILD'
NATIVE_SIZE = (1920, 1080)


class SourceOverlay(importlib.abc.MetaPathFinder):
    """指定モジュールだけ worktree のファイルから読む。"""

    def __init__(self, root: Path, names: list[str]) -> None:
        self.files = {name.removesuffix('.py').replace('/', '.'): root/name for name in names}

    def find_spec(self, fullname: str, path: object, target: object = None) -> object:
        source = self.files.get(fullname)
        return importlib.util.spec_from_file_location(fullname, source) if source else None


class PostImportPatch(importlib.abc.MetaPathFinder):
    """対象モジュールの実行直後に差し替え関数を呼ぶ。読込み元は他の finder に任せる。"""

    def __init__(self, patches: dict[str, Callable[[Any], None]]) -> None:
        self.patches = patches

    def find_spec(self, fullname: str, path: object, target: object = None) -> object:
        if fullname not in self.patches:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        execute, patch = spec.loader.exec_module, self.patches[fullname]

        def exec_module(module: Any) -> None:
            execute(module)
            patch(module)
        spec.loader.exec_module = exec_module
        return spec


def cropped_source(plan: dict) -> Callable[[Any], None]:
    def patch(module: Any) -> None:
        import cv2
        from dataclasses import replace
        from src.phase_j.dshow_color import correct_601_as_709
        crop = plan.get('crop')   # None = 切り抜かない (1920x1080 への拡大は VideoFileSource が済ませている)

        class CroppedSource(module.VideoFileSource):
            def __iter__(self) -> Any:
                for frame in super().__iter__():
                    image = frame.image
                    if crop is not None:
                        width, height = crop
                        image = cv2.resize(image[:height, :width], NATIVE_SIZE, interpolation=cv2.INTER_AREA)
                    if plan.get('color_correction') == '601to709':
                        image = correct_601_as_709(image)
                    yield replace(frame, media_sec=frame.media_sec+plan['offset_sec'], image=image)
        module.VideoFileSource = CroppedSource
        mark(plan, 'video_source')
    return patch


def recognition_reference(plan: dict) -> Callable[[Any], None]:
    def patch(module: Any) -> None:
        original = module.recognition_config

        def recognition_config(config: dict, live_input: bool) -> dict:
            return dict(original(config, live_input), **plan['recognition_overrides'])
        module.recognition_config = recognition_config
        mark(plan, 'recognition_config')
    return patch


def launcher_paths(plan: dict) -> Callable[[Any], None]:
    def patch(module: Any) -> None:
        module.APP_ROOT = Path(plan['app'])  # 差し替えた launcher.py でも配布 app を基準にする
        original = module.build_pipeline_config

        def build_pipeline_config(user: Any) -> dict:
            return dict(original(user), realtime=plan['realtime'])
        module.build_pipeline_config = build_pipeline_config
        mark(plan, 'launcher')
    return patch


def mark(plan: dict, name: str) -> None:
    """差し替えた process と箇所を受領票へ追記する (0件を成功と読まないため)。"""
    with open(plan['patch_log'], 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(pid=os.getpid(), patch=name))+'\n')


def install(plan: dict) -> None:
    # obs_runtime_bootstrap.py と同じく、実行系の ._pth が足す旧 app を外して配布 app を先頭に置く。
    old_app = (Path(sys.executable).resolve().parent.parent/'app').resolve()
    sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != old_app]
    sys.path.insert(0, plan['app'])
    sys.meta_path.insert(0, SourceOverlay(Path(plan['source_root']), plan['source_modules']))
    sys.meta_path.insert(0, PostImportPatch({
        'src.phase_j.live_process': cropped_source(plan),
        'src.phase_j.live_speed': recognition_reference(plan),
        'src.phase_j.launcher': launcher_paths(plan)}))


if ENV_PLAN in os.environ:
    install(json.loads(os.environ[ENV_PLAN]))

if __name__ == '__main__':
    if ENV_PLAN not in os.environ:
        raise SystemExit('run_live_x_replay.py からだけ起動する')
    from src.phase_j.launcher import main
    raise SystemExit(main(sys.argv[1:]))
