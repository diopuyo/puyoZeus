"""live_recog_speed_py (認識ループの Python 側の出力不変の高速化) の検証台。診断用で、製品の既定値は変えない。

実 live の経路を再現する: 取り込み画素 → 色補正 (601to709) → live_process.recognize (DeviceSession 経由)。
配布 (24d の C2) と同じ環境変数は、配布の起動口と同じ parse_args から作る (手書きしない)。
1 条件 = 1 プロセスで直列に回し、コマごとの認識結果・公開値・B3 の観測を別々のハッシュにして保存する。

  prepare : 配布 app (基点) を複製し、変更したファイルだけ重ねた「候補 app」を作る
  run     : 1 条件を実行 (コマ別ハッシュ・認識時間・使い回しの件数を保存)
  compare : 基準と候補をコマごと・欄ごとに全件照合 (母数つき、違いの最初のコマも出す)
  speed   : 保存済みの認識時間 (暖機を除く) の平均・P50・P95
  audit-images : 発火前画像の分類を、従来 (全画面保持・1 セルずつ) と新 (盤面だけ保持・一括) で全セル比較
  selftest : 比較器自身の健全性 (1 セルの違い・得点 1 の違いを必ず検出すること)

測定器の健全性 (fail-silent 防止): selftest と、同じ条件を 2 回回す control (違い 0 であること)、
別区間を比べる negative control (違いが検出されること) を併用する。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = Path('C:/PuyoLive_OBS_20261025d/PuyoLive')
BUNDLE_APP = BUNDLE_ROOT / 'app'
BUNDLE_PYTHON = BUNDLE_ROOT / 'python/python.exe'
WORK = Path(_public_dev_text('<work>/recog_speed_py_20261025'))
BASE_COMMIT = 'ea7a45b2'   # 配布 app (24d) の src の元になったコミット (src は ac76927c と同一)
CONFIG_DIR = WORK / 'config'   # 24d の C2 走行の pipeline_config / recognition_config / puyo_live (凍結した写し)
RUNS = WORK / 'runs'
DEFAULT_VIDEO = 'C:/puyo1_tools/soak/media/video_d04.mp4'
SHAPE = (1080, 1920, 3)
FRAME_BYTES = SHAPE[0] * SHAPE[1] * SHAPE[2]
DECODER = Path('C:/puyo1_tools/soak/vcam_venv/Scripts/python.exe')
FPS = 30.0
MS = 1000.0
WARMUP = 60
BELOW_NORMAL = 0x4000
MIN_FREE_BYTES = 6 * 1024 ** 3
HOLD_FILE = Path(_public_dev_text('<work>/HOLD_FOR_OBS'))
# 実行ごとに変わる (出力ではなく計時の) 値。比較の前に取り除く。control で違いが出たらここへ名前で足す。
TIMING_KEYS = frozenset({'image_compute_ms'})
NOTICE_FIELDS = ('frame', 't_sec', 'result_bytes', 'pipeline', 'dropped_before')
B3_PARTS = ('row_core', 'v49f', 'ama', 'events', 'queues', 'epoch')
SIDES = ('p1', 'p2')
BOARD_FIELDS = ('raw_cnn_board', 'cnn_board', 'confirmed_board')
BOARD_SHAPE = (13, 6)
# 24d の OBS 試験の入口 (start_obs_trial.py) が設定する環境。PUYO_OBS_DEEP_TIMING は配布の既定が OFF なので入れない
# (必要なら --env PUYO_OBS_DEEP_TIMING=1 と --stage-timing で入れ、費用を ON/OFF の差で測る)。
TRIAL_ENTRY_ENV = {'PUYO_OBS_CPU_ISOLATION': '0', 'PUYO_OBS_AMA_IDLE': '0', 'PUYO_OBS_FAST_AUDIT': '1',
                   'PUYO_OBS_EXACT_SEED': '1', 'PUYO_OBS_PREPARE_EXACT': '1'}


# ----------------------------------------------------------------------------
# 安全弁・共通
# ----------------------------------------------------------------------------

def free_bytes() -> int:
    """依存を足さず Windows の空き物理メモリを返す。"""
    import ctypes

    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('total', 'available', 'page_total', 'page_available', 'virtual', 'virtual_available', 'extended')]
    memory = Memory()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    return int(memory.available)


def guard() -> None:
    """開始前と各コマ境界で、利用者の負荷制限 (OBS 試験中の HOLD・空き 6GiB) を守る。"""
    if HOLD_FILE.exists():
        raise RuntimeError('HOLD_FOR_OBS が存在するため停止')
    if free_bytes() < MIN_FREE_BYTES:
        raise RuntimeError('空きメモリが 6GiB 未満のため停止')


def thread_cycles() -> int:
    """呼んだスレッドが使った CPU サイクル数 (QueryThreadCycleTime)。他 process に奪われた時間を含まない。

    壁時計は PC1 が他の作業 (OBS・別の再生など) と取り合うと +-3ms 揺れ、項目の効果 (0.5〜2ms) より大きい。
    サイクル数は取り合いの時間を含まないので、同じ作業量なら揺れにくい (周波数が変わっても作業量は同じ)。
    """
    import ctypes
    kernel32 = ctypes.windll.kernel32
    # 擬似ハンドル (-2) を 64 ビットで渡すため、型を明示する (既定の 32 ビット int だと無効なハンドルになる)。
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    kernel32.QueryThreadCycleTime.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
    value = ctypes.c_ulonglong(0)
    if not kernel32.QueryThreadCycleTime(kernel32.GetCurrentThread(), ctypes.byref(value)):
        raise ctypes.WinError()
    return int(value.value)


def strip_timing(value: Any) -> Any:
    """計時の値 (TIMING_KEYS) を再帰的に取り除いた複製。辞書の順序・型は保つ。"""
    if isinstance(value, dict):
        return type(value)((k, strip_timing(v)) for k, v in value.items() if k not in TIMING_KEYS)
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    if isinstance(value, tuple):
        return tuple(strip_timing(v) for v in value)
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def b3_parts(b3: Any) -> dict[str, str]:
    """B3 の観測 (row, events, queues, epoch) を、原因の場所が分かる単位のハッシュにする。"""
    if b3 is None:
        return {name: digest(None) for name in B3_PARTS}
    row, events, queues, epoch = b3
    core = {k: v for k, v in row.items() if k not in ('_v49f', '_ama')}
    return dict(row_core=digest(strip_timing(core)), v49f=digest(strip_timing(row.get('_v49f'))),
                ama=digest(strip_timing(row.get('_ama'))), events=digest(strip_timing(events)),
                queues=digest(strip_timing(queues)), epoch=digest(epoch))


def notice_record(notice: Any) -> dict[str, str]:
    """通知 1 件のコマ別ハッシュ。時刻 (captured_at・recognized_at) は壁時計なので入れない。"""
    record = {name: digest(getattr(notice, name)) for name in NOTICE_FIELDS}
    for name, value in b3_parts(notice.b3).items():
        record['b3.' + name] = value
    return record


# ----------------------------------------------------------------------------
# prepare: 候補 app
# ----------------------------------------------------------------------------

def changed_files() -> list[str]:
    """基点 (BASE_COMMIT = 配布 app の元のコミット) から、作業ツリーまでに変えた・足した .py (候補 app へ重ねるもの)。

    HEAD との差ではなく基点との差にする (コミット後も同じ候補 app を作れるように)。
    """
    git = ['git', '-C', str(ROOT)]
    changed = subprocess.run([*git, 'diff', '--name-only', BASE_COMMIT], capture_output=True, text=True, check=True)
    added = subprocess.run([*git, 'ls-files', '--others', '--exclude-standard'], capture_output=True, text=True,
                           check=True)
    names = [n for n in (changed.stdout + added.stdout).split() if n.endswith('.py')]
    return sorted(n for n in set(names) if n.startswith(('src/', 'scripts/')))


def prepare_app(dest: Path) -> dict[str, Any]:
    """配布 app (基点) を複製し、変更したファイルだけ作業ツリーの版で上書きする。"""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(BUNDLE_APP, dest, ignore=shutil.ignore_patterns('__pycache__'))
    overlaid = []
    for name in changed_files():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        overlaid.append(dict(path=name, sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest()))
    record = dict(dest=str(dest), source_head=subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                  capture_output=True, text=True, check=True).stdout.strip(), overlaid=overlaid)
    (dest.parent / (dest.name + '.overlay.json')).write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                                             encoding='utf-8')
    return record


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------

def decode_frames(video: str, start: float, count: int) -> Any:
    """動画を別 process で逐次デコードして 1080p BGR を返す (全画像を貯めない)。

    配布 python の cv2 は動画を開けない (ffmpeg 無し) ので、専用デコーダ (動画を開ける cv2) の
    process へ生画素を流してもらう。既存の品質門 (live_integ_gate.py) と同じ方式。
    """
    import numpy as np
    process = subprocess.Popen([str(DECODER), '-B', str(Path(__file__).resolve()), 'decode',
                                video, str(start), str(count)], stdout=subprocess.PIPE,
                               creationflags=BELOW_NORMAL)
    try:
        for _ in range(count):
            raw = process.stdout.read(FRAME_BYTES)
            if len(raw) != FRAME_BYTES:
                raise RuntimeError('デコードが指定枚数より前に終了')
            yield np.frombuffer(raw, dtype=np.uint8).reshape(SHAPE)
    finally:
        process.stdout.close()
        process.terminate()
        process.wait()


def decode_main(video: str, start: float, count: int) -> None:
    """専用デコーダ側。動画を開き、1080p BGR の生画素を標準出力へ流す。"""
    import cv2
    import msvcrt
    cv2.setNumThreads(1)
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    capture = cv2.VideoCapture(video)
    capture.set(cv2.CAP_PROP_POS_MSEC, start * MS)
    try:
        for _ in range(count):
            ok, image = capture.read()
            if not ok:
                break
            if image.shape != SHAPE:
                image = cv2.resize(image, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_AREA)
            sys.stdout.buffer.write(image.tobytes())
    finally:
        capture.release()


def distribution_env(output: Path, extra: list[str]) -> None:
    """配布の起動口 (run_live_pipeline の parse_args) と同じ環境変数を作り、追加の指定で上書きする。"""
    config = CONFIG_DIR / 'pipeline_config.json'
    sys.argv = ['run', '--config', str(config), '--output', str(output), '--input-config', str(CONFIG_DIR / 'device.json')]
    from scripts.run_live_pipeline_20260928 import parse_args
    parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    for key, value in TRIAL_ENTRY_ENV.items():   # start_obs_trial.py が認識 process へ渡す環境のうち、詳細計装以外
        os.environ.setdefault(key, value)
    if (Path('models/onnx/index.json')).is_file():
        os.environ['PUYO_CNN_BACKEND'] = 'onnx'
    for item in extra:
        key, value = item.split('=', 1)
        os.environ[key] = value


def build_session(output: Path) -> Any:
    """配布の認識 process と同じ順で認識器と DeviceSession を作る (乱数の種も同じ)。"""
    from src.recognition_pipeline import RecognitionPipeline
    from scripts.run_e3_exchange_eval_20260926 import SEED
    import numpy as np
    import torch
    from src.phase_j.live_cpu import apply_runtime
    apply_runtime('recognition')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    pipe = RecognitionPipeline.load_default(**json.loads((CONFIG_DIR / 'recognition_config.json').read_text('utf-8')))
    if os.environ.get('PUYO_OBS_PREPARE_EXACT') == '1':
        from src.phase_j.live_obs_prepare import prepare_recognition
        prepare_recognition(pipe)
    from src.phase_j.live_device import DeviceConfig
    from src.phase_j.live_device_session import DeviceSession
    from src.phase_j.live_process import configure_media_tick
    device = CONFIG_DIR / 'device.json'
    session = DeviceSession(pipe, device, 1e9, _NullQueue())
    configure_media_tick(session, 1)
    session.switch(DeviceConfig.load(device))
    from src.phase_j.live_b3 import prepare_capture
    prepare_capture(session)
    return session


class _NullQueue:
    def put(self, *_: Any, **__: Any) -> None:
        return None


def run_condition(args: argparse.Namespace) -> None:
    guard()
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
    app = Path(args.app).resolve()
    sys.path.insert(0, str(app))
    os.chdir(app)
    out = RUNS / args.tag
    out.mkdir(parents=True, exist_ok=True)
    os.environ['PUYO_CALIBRATION_DIR'] = str(out / 'calibration')
    distribution_env(out, args.env)
    session = build_session(out)
    if args.coverage:
        install_coverage()
    if args.sync_pair:
        install_sync_pair()
    with ExitStack() as storage:
        install_timing(storage, session, out, args.stage_timing)
        records, times, cycles = drive(session, args)
    stats = collect_stats()
    (out / 'records.pickle').write_bytes(pickle.dumps(dict(records=records, times=times, cycles=cycles, stats=stats)))
    (out / 'condition.json').write_text(json.dumps(dict(vars(args), frames_done=len(records), stats=stats,
        env={k: v for k, v in os.environ.items() if k.startswith('PUYO_')}, python=sys.version), default=str,
        ensure_ascii=False, indent=1), encoding='utf-8')
    print(args.tag, 'frames', len(records), 'stats', stats, flush=True)


def install_timing(storage: ExitStack, session: Any, out: Path, stage_timing: bool) -> None:
    """段別計装 (live_stage_timing) と詳細計装 (PUYO_OBS_DEEP_TIMING) を、配布の認識 process と同じ順で入れる。

    どちらも既定では入れない (配布の既定は OFF)。C2 の試験は両方 ON だったので、その費用を
    ON/OFF の差で測るために使う。
    """
    if not stage_timing:
        return
    from src.phase_j import live_stage_timing as timing
    if os.environ.get('PUYO_OBS_DEEP_TIMING') == '1':
        from src.phase_j.live_obs_recognition_timing import install
        install(session.pipe)
    timing.install(storage, out / 'spool', 'recognition')


def drive(session: Any, args: argparse.Namespace) -> tuple[list, list, list]:
    """フレームを順に投入し、コマ別ハッシュ・認識時間 (ms)・認識に使ったスレッドのサイクル数を返す。"""
    from src.phase_j import live_process
    from src.phase_j.dshow_color import correct_601_as_709
    from src.phase_j.live_source import CapturedFrame
    first = round(args.start * FPS)
    records, times, cycles = [], [], []
    for position, raw in enumerate(decode_frames(args.video, args.start, args.frames)):
        guard()
        image = correct_601_as_709(raw)
        index = first + position
        now = time.perf_counter()
        frame = CapturedFrame(index, index / FPS, now, now, image)
        cycles_before = thread_cycles()
        started = time.perf_counter()
        notice = live_process.recognize(session, frame)
        times.append((time.perf_counter() - started) * MS)
        cycles.append(thread_cycles() - cycles_before)
        if args.stage_timing:
            from src.phase_j.live_stage_timing import finish
            finish(held=False)
        records.append(notice_record(notice))
        if args.coverage:
            note_coverage(notice)
        del notice, image
    return records, times, cycles


COVERAGE: dict[str, int] = {}
# 終了の組の照合を決定的に再現するときの周期 (コマ数)。実機は 0.5 秒 = 30 コマ/秒で約 15 コマごと。
SYNC_PAIR_PERIOD_FRAMES = 15


def install_sync_pair() -> None:
    """終了の組の非同期照合を、コマ数で決まる決定的な照合に置き換える (全条件で同じ。出力の比較専用)。

    実機の照合は別 thread で 0.5 秒ごとの壁時計で動くので、同じ画素を入れても結果の採用が 1 コマ前後して
    全行一致が (基点どうしでも) 取れない (control の off と off2 でも違いが出ることで確認済み)。
    照合の中身 (_evaluate) と採用の規則 (前コマまでの完了を source_t <= t_sec で採用) は元のまま、
    「15 コマごとに評価し、次のコマで採用」へ固定する。
    """
    from src.async_match_end_pair import AsyncMatchEndPairDetector

    def update(self: Any, frame_bgr: Any, t_sec: float) -> bool:
        completed, self._completed = self._completed, None
        if completed is not None and completed[0] <= t_sec:
            source_t, result = completed
            self.last_detection, self.last_detection_t = result, source_t
            if result.detected:
                if self.lock_started_t is None:
                    self.lock_started_t = source_t
                self._last_detected_t = source_t
        counter = getattr(self, '_sync_counter', 0)
        self._sync_counter = counter + 1
        if counter % SYNC_PAIR_PERIOD_FRAMES == 0:
            from src.match_end_pair import PAIR_ROIS
            crops = {key: frame_bgr[y:y + h, x:x + w].copy() for key, (x, y, w, h) in PAIR_ROIS.items()}
            self._completed = (t_sec, self._evaluate(crops, t_sec))
        return self.is_locked(t_sec)
    AsyncMatchEndPairDetector.update = update


def install_coverage() -> None:
    """窓に対象の事象が入っているかを数える (出力は変えない)。発火前画像の分類・追加の呼出回数。"""
    from src.phase_j.live_v49f_images import LiveImages
    classify, append = LiveImages.classify, LiveImages.append

    def counted_classify(self: Any, image: Any, stamp: float, side: int) -> Any:
        COVERAGE['live_images_classify_calls'] = COVERAGE.get('live_images_classify_calls', 0) + 1
        return classify(self, image, stamp, side)

    def counted_append(self: Any, stamp: float, image: Any) -> Any:
        COVERAGE['live_images_append_calls'] = COVERAGE.get('live_images_append_calls', 0) + 1
        return append(self, stamp, image)
    LiveImages.classify, LiveImages.append = counted_classify, counted_append


def note_coverage(notice: Any) -> None:
    """通知から、試合中・連鎖中・B3 の起点が出たコマ数を数える。"""
    result = pickle.loads(notice.result_bytes)
    COVERAGE['frames'] = COVERAGE.get('frames', 0) + 1
    COVERAGE['frames_in_match'] = COVERAGE.get('frames_in_match', 0) + bool(result.is_match_active)
    COVERAGE['frames_chain_event'] = COVERAGE.get('frames_chain_event', 0) + any(
        getattr(side, 'chain_event', None) is not None for side in (result.p1, result.p2))
    packet = (notice.b3[0] or {}).get('_v49f') if notice.b3 else None
    COVERAGE['frames_v49f_origins'] = COVERAGE.get('frames_v49f_origins', 0) + bool(packet and packet.get('origins'))
    COVERAGE['frames_match_end_locked'] = COVERAGE.get('frames_match_end_locked', 0) + bool(
        getattr(result, 'match_end_locked', False))


def collect_stats() -> dict[str, int]:
    """候補側の使い回し・監査の件数 (基点にはモジュールが無いので空)。"""
    try:
        from src.recog_speed_py import STATS
    except ImportError:
        return dict(COVERAGE)
    stats = dict(STATS, **{'coverage.' + k: v for k, v in COVERAGE.items()})
    from src.live_board_vectorized import STATS as VECTORIZED_STATS   # 一括経路 (native) が通った件数。診断のみ
    return dict(stats, **{'vectorized.' + k: v for k, v in VECTORIZED_STATS.items()})


# ----------------------------------------------------------------------------
# compare / speed
# ----------------------------------------------------------------------------

def load_run(tag: str) -> dict[str, Any]:
    return pickle.loads((RUNS / tag / 'records.pickle').read_bytes())


def compare_runs(base_tag: str, cand_tag: str) -> dict[str, Any]:
    """欄ごとに全コマを照合する。母数 (比べたコマ数・欄数) と、違いの最初のコマを残す。"""
    base, cand = load_run(base_tag), load_run(cand_tag)
    count = min(len(base['records']), len(cand['records']))
    fields = list(base['records'][0]) if count else []
    mismatch = {name: [i for i in range(count) if base['records'][i][name] != cand['records'][i][name]]
                for name in fields}
    any_frames = sorted({i for indexes in mismatch.values() for i in indexes})
    return dict(base=base_tag, candidate=cand_tag, frames_base=len(base['records']),
                frames_candidate=len(cand['records']), frames_compared=count, fields_compared=len(fields),
                values_compared=count * len(fields),
                field_mismatch_frames={name: len(v) for name, v in mismatch.items()},
                frames_any_mismatch=len(any_frames), first_mismatch_frames=any_frames[:10],
                candidate_stats=cand.get('stats', {}))


def summarize(values: list[float]) -> dict[str, float]:
    import numpy as np
    array = np.array(values)
    return dict(n=len(values), mean=float(array.mean()), p50=float(np.percentile(array, 50)),
                p95=float(np.percentile(array, 95)))


def speed_runs(tags: list[str]) -> dict[str, Any]:
    """暖機 WARMUP コマと終端 1 コマを除いた認識時間 (ms、割った単位 = 認識 1 コマの壁時計) と、
    同じコマのスレッドサイクル数 (Mcycles、割った単位 = 認識 1 コマ)。"""
    out: dict[str, Any] = {}
    for tag in tags:
        run = load_run(tag)
        cycles = [c / 1e6 for c in run.get('cycles', [])][WARMUP:-1]
        out[tag] = dict(wall_ms=summarize(run['times'][WARMUP:-1]), mcycles=summarize(cycles) if cycles else None)
    return out


def paired_effect(candidate: str, before: str, after: str) -> dict[str, Any]:
    """候補 1 本と、前後の A (何も効かない) 2 本の平均を、同じコマどうしで引いた差 (負 = 速くなった)。

    コマごとの差の平均と中央値 (外れ値に強い) を、壁時計 (ms) とサイクル数 (Mcycles) で出す。
    A 前後の差も出す (A どうしの差 = 測定の揺れの大きさ。候補の効果がこれより小さければ判定できない)。
    """
    import numpy as np
    runs = [load_run(t) for t in (candidate, before, after)]
    out: dict[str, Any] = {}
    for name, scale in (('times', 1.0), ('cycles', 1e-6)):
        if any(name not in run or not run[name] for run in runs):
            continue
        cand, a1, a2 = (np.array(run[name][WARMUP:-1], dtype=float) * scale for run in runs)
        n = min(len(cand), len(a1), len(a2))
        diff = cand[:n] - (a1[:n] + a2[:n]) / 2
        out[name] = dict(n=int(n), mean=float(diff.mean()), median=float(np.median(diff)),
                         a_noise_mean=float((a2[:n] - a1[:n]).mean()))
    return out


# ----------------------------------------------------------------------------
# selftest (比較器の健全性)
# ----------------------------------------------------------------------------

def selftest() -> dict[str, bool]:
    """1 セルの違い・得点 1 の違い・計時だけの違いを、比較器が正しく扱うこと。"""
    import numpy as np
    grid = np.zeros(BOARD_SHAPE, dtype=np.int8)
    changed = grid.copy()
    changed[12, 5] = 1
    row = dict(frame=1, sides=[dict(score=100)], _v49f=dict(sources=[dict(board=[0, 1], image_compute_ms=1.5)]))
    row_score = dict(row, sides=[dict(score=101)])
    row_timing = dict(row, _v49f=dict(sources=[dict(board=[0, 1], image_compute_ms=9.9)]))
    wrap = lambda value: (value, (), (), (0, True))  # noqa: E731
    return dict(
        cell_difference_detected=digest(grid) != digest(changed),
        score_difference_detected=b3_parts(wrap(row))['row_core'] != b3_parts(wrap(row_score))['row_core'],
        timing_only_difference_ignored=b3_parts(wrap(row)) == b3_parts(wrap(row_timing)),
        v49f_board_difference_detected=b3_parts(wrap(row))['v49f'] != b3_parts(wrap(dict(
            row, _v49f=dict(sources=[dict(board=[0, 2], image_compute_ms=1.5)]))))['v49f'])


# ----------------------------------------------------------------------------
# audit-images
# ----------------------------------------------------------------------------

def audit_images(args: argparse.Namespace) -> dict[str, Any]:
    """発火前画像の分類を、従来と新 (image_crops + image_batch) で、実コマの全セルについて比べる。"""
    guard()
    app = Path(args.app).resolve()
    sys.path.insert(0, str(app))
    os.chdir(app)
    os.environ.update(PUYO_OBS_PREPARE_EXACT='1', PUYO_LIVE_NATIVE_HSV_MAIN='1', PYTHONDONTWRITEBYTECODE='1')
    from src.recog_speed_py import ENV_FLAG, ENV_ITEMS, ITEM_IMAGE_BATCH, ITEM_IMAGE_CROPS
    from src.phase_j.live_obs_prepare import create_v49f_capture
    from src.phase_j.dshow_color import correct_601_as_709
    legacy, fast = create_v49f_capture().images, create_v49f_capture().images
    compared = differ = 0
    for position, raw in enumerate(decode_frames(args.video, args.start, args.frames)):
        image = correct_601_as_709(raw)
        stamp = position / FPS
        os.environ.pop(ENV_FLAG, None)
        legacy.append(stamp, image)
        os.environ.update({ENV_FLAG: '1', ENV_ITEMS: ITEM_IMAGE_CROPS + ',' + ITEM_IMAGE_BATCH})
        fast.append(stamp, image)
        for side in (0, 1):
            os.environ.pop(ENV_FLAG, None)
            expected = [strip_timing(b) for b in legacy.boards(stamp, stamp, side, (0.,))]
            os.environ.update({ENV_FLAG: '1', ENV_ITEMS: ITEM_IMAGE_CROPS + ',' + ITEM_IMAGE_BATCH})
            actual = [strip_timing(b) for b in fast.boards(stamp, stamp, side, (0.,))]
            compared += sum(len(b['board']) for b in expected)
            differ += sum(a != e for a, e in zip(actual, expected))
    from src.live_board_vectorized import STATS as VECTORIZED_STATS
    return dict(frames=args.frames, sides=2, cells_compared=compared, boards_differ=differ,
                native_median_classifier=bool(fast.classifiers[0]._enable_native_hsv),
                native_batch_calls=int(VECTORIZED_STATS.get('classify_native_calls', 0)),
                per_patch_cells=int(VECTORIZED_STATS.get('classify_per_patch_cells', 0)))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--dest', type=Path, default=WORK / 'app_new')
    run = sub.add_parser('run')
    run.add_argument('--tag', required=True)
    run.add_argument('--app', type=Path, default=BUNDLE_APP)
    run.add_argument('--video', default=DEFAULT_VIDEO)
    run.add_argument('--start', type=float, default=650.0)
    run.add_argument('--frames', type=int, default=900)
    run.add_argument('--env', action='append', default=[])
    run.add_argument('--stage-timing', action='store_true', help='段別計装 (live_stage_timing) を入れる')
    run.add_argument('--sync-pair', action='store_true',
                     help='終了の組の照合を決定的な周期へ固定する (出力の全行一致の比較用。速度の測定では付けない)')
    run.add_argument('--coverage', action='store_true', help='窓に対象の事象が入っているかを数える (速度の測定では付けない)')
    cmp_ = sub.add_parser('compare')
    cmp_.add_argument('base')
    cmp_.add_argument('candidates', nargs='+')
    cmp_.add_argument('--out', type=Path)
    spd = sub.add_parser('speed')
    spd.add_argument('tags', nargs='+')
    pair = sub.add_parser('paired')
    pair.add_argument('triples', nargs='+', help='候補:前のA:後のA を空白区切りで並べる')
    aud = sub.add_parser('audit-images')
    aud.add_argument('--app', type=Path, default=WORK / 'app_new')
    aud.add_argument('--video', default=DEFAULT_VIDEO)
    aud.add_argument('--start', type=float, default=650.0)
    aud.add_argument('--frames', type=int, default=300)
    sub.add_parser('selftest')
    return parser.parse_args()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == 'decode':
        decode_main(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]))
        return
    args = parse()
    if args.command == 'prepare':
        print(json.dumps(prepare_app(args.dest), ensure_ascii=False, indent=1))
    elif args.command == 'run':
        run_condition(args)
    elif args.command == 'compare':
        rows = [compare_runs(args.base, tag) for tag in args.candidates]
        text = json.dumps(rows, ensure_ascii=False, indent=1)
        if args.out:
            args.out.write_text(text, encoding='utf-8')
        print(text)
    elif args.command == 'speed':
        print(json.dumps(speed_runs(args.tags), ensure_ascii=False, indent=1))
    elif args.command == 'paired':
        print(json.dumps({t: paired_effect(*t.split(':')) for t in args.triples}, ensure_ascii=False, indent=1))
    elif args.command == 'audit-images':
        print(json.dumps(audit_images(args), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(selftest(), indent=1))


if __name__ == '__main__':
    main()
