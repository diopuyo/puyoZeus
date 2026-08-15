"""指摘14修正2フラグ (--resolved-live-defender-strict / --resolved-kill-override)
の全域バックテスト (2026-08-15、coordinator依頼)。

## 目的
1場面 (絶対194.53-201秒、review_demo_2026-08-12.mp4) では退行消滅を確認済み
(scripts/_diag_issue14_flags_ab_v2_2026-08-15.py 等)。本スクリプトは
feedback_overfitting_awareness_2026-08-04 (「5シーン合格+全域無悪化」で
初めて合格) に基づき、**判定・表示層の出力分布**への広域影響を測る。

## 既存バックテストとの違い (重要)
scripts/_backtest_placement_override_full_2026-08-15.py は **認識**
(churn/重力違反/OJAMA_FALL滞在) を測るもの。本スクリプトは
scripts.visualize_advantage_overlay.generate() を render=False で通し、
**disp_adv (表示に出る有利不利値) の時系列そのもの**を OFF/ON で比較する。
設計 (代表サンプル方式・並列ドライバ・read-only計装) は上記スクリプトを
踏襲するが、指標は完全に別物。

## 計装 (read-only, production コード無変更)
`ResolvedExchangeTracker.update` はモジュール内に決着ホールドの活性状態
(`self._active`) を保持するが、呼出元 generate() には (is_active,
just_deactivated) の2値しか公開しない。決着ホールドの発生回数・継続時間を
フレーム単位で追うため、`ov.ResolvedExchangeTracker.update` を薄いラッパーへ
差し替え (このプロセス内限定、production コード自体は無変更)、戻り値と
呼出時刻 (t_sec kwarg) を副作用なく記録してから元のメソッドへ委譲する
(_backtest_placement_override_full_2026-08-15.py の `_process_side_lean`
差し替えと同じ方式)。

## 構成 (2構成)
- OFF: 指摘14 2フラグとも False (production_config.py 採用前の挙動)
- ON:  指摘14 2フラグとも True (2026-08-15 採用登録済みの本番構成)
両方とも共通の土台は `_gen_demo_final3_2026-08-15.sh` 実写生成コマンドと
同一 (resolved-exchange-eval/decisive-amplify/live-defender 込みの
「現在実際に使われている最良構成」。**この土台自体は
src/production_config.py の ADVANTAGE_ADOPTED に未登録という食い違いが
判明したが [2026-08-15 本バックテスト作成時の副次発見、別途要フォロー]、
本バックテストの目的である「指摘14の2フラグ単体の広域影響」を測るには
影響しない=土台をOFF/ON双方に同一に与えているため)。

## 位置づけ (正直な開示)
フル尺ではなく **代表サンプル** (動画あたり 中盤/終盤 2地点 × CHUNK_SEC 秒。
序盤は決着ホールドが疎という判断で割愛、詳細は N_CHUNKS_PER_VIDEO 節参照)。
決着ホールド (両側同時発火) 自体が疎な事象のため、代表サンプルでは
「発生ゼロ」の区間も相当数出ることを許容する (発生した区間の分布で
判断する設計)。

## 使い方
    # ドライバ (全ジョブ実行 + 収集後に集計)
    PYTHONPATH=. ./venv/bin/python -m \\
        scripts._backtest_issue14_flags_2026-08-15

    # 集計のみ (収集済み npz から再集計、チェックポイント再開用)
    PYTHONPATH=. ./venv/bin/python -m \\
        scripts._backtest_issue14_flags_2026-08-15 --aggregate-only

    # 単一ジョブ (内部で driver が subprocess 起動する用、直接使わない)
    PYTHONPATH=. ./venv/bin/python -m \\
        scripts._backtest_issue14_flags_2026-08-15 --worker \\
        --video <path> --config off --start-sec 100 --chunk-sec 90 \\
        --out-npz <path>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ============================
# 定数 (マジックナンバー禁止規約)
# ============================

VIDEO_DIR: Path = Path.home() / "frames"
OUT_ROOT: Path = _ROOT / "data" / "verify" / "backtest_issue14_2026-08-15"
LOG_ROOT: Path = _ROOT / "logs" / "backtest_issue14_2026-08-15"
PYTHON_BIN: Path = _ROOT / "venv" / "bin" / "python"

# 手元実物動画 (WSL ~/frames/ 実物)。指摘14の調整に使った
# data/frames/review_demo_2026-08-12.mp4 とは別資産を優先選定する
# (過学習防止=調整に使った動画で採点しない)。
# video_c15/c19 は placement_override 全域バックテストで既知のファイル破損
# (production_config.py の該当コメント参照、collect_lean 側の症状だが
# 疑わしいため本バックテストでも除外)。video_c96/c109 は330分/226分の
# 複数試合連結ファイルで計算コストが桁違いのため今回は対象外
# (「測れていないこと」として報告に明記)。
VIDEO_FILENAMES: tuple[str, ...] = (
    "video_c10.mp4", "video_c11.mp4", "video_c12.mp4", "video_c13.mp4",
    "video_c14.mp4", "video_c16.mp4", "video_c17.mp4", "video_c18.mp4",
    "video_c20.mp4", "video_c21.mp4", "video_c22.mp4", "video_c23.mp4",
)

# 1地点あたりの処理長 [秒]。calib実測2回:
# (1) 単独実行 (10秒content): ≈230秒 (23秒/content秒)
# (2) 10並列 (BLAS/OMP=1スレッド固定後、10秒content): ≈200秒、15秒content:
#     ≈500秒 (≈35秒/content秒、単独比1.5倍程度の残存コンテンション)
# **初回1スレッド未固定での10並列は20分経過して1ジョブも完了せず**
# (BLAS/OMPスレッドのオーバーサブスクリプションが原因、_SINGLE_THREAD_ENV
# 参照)。35秒/content秒×10並列前提で 動画12本×2地点×2構成=48ジョブ
# ×CHUNK_SEC=30秒 ≈ 48/10並列 × 17.5分 ≈ 1.4時間の見積り
# (「フル尺は非現実的」で代表サンプルに絞る判断は
# _backtest_placement_override_full_2026-08-15.py と同じ)。
CHUNK_SEC: float = 30.0
# 動画1本あたりの地点数。決着ホールド (両側同時発火) は疎な事象で中盤〜
# 終盤に出やすいため、序盤を割愛し中盤/終盤の2地点に絞る (工数との兼ね合い、
# 「測れていないこと」として報告に明記する)。
N_CHUNKS_PER_VIDEO: int = 2
CHUNK_OFFSET_FRACTIONS: tuple[float, ...] = (0.35, 0.75)
# 同時実行プロセス数 (16コア中、他ジョブ (フルpytest等) との共存を考慮し10)
MAX_PARALLEL_WORKERS: int = 10

CONFIG_OFF: str = "off"  # 指摘14 2フラグとも False (採用前)
CONFIG_ON: str = "on"    # 指摘14 2フラグとも True (2026-08-15 採用構成)

# 表示勝率の極端値しきい値 (指摘の意図=致死上書きは想定通り、非致死場面での
# 増加が無いかを見る)
EXTREME_P1_HIGH: float = 0.95
EXTREME_P1_LOW: float = 0.05

# 同一時刻とみなす丸め桁数 (OFF/ON 両histのt突合わせ用、fi/fpsの浮動小数
# 誤差吸収)
T_MATCH_DECIMALS: int = 3
# フレーム抽出時、動画の fps が読めない場合のフォールバック値
DEFAULT_FPS_FALLBACK: float = 30.0


def common_overlay_kwargs() -> dict[str, object]:
    """OFF/ON 双方に共通の土台 kwargs を返す。

    scripts/_gen_demo_final3_2026-08-15.sh (2026-08-15 時点で実際に使われて
    いる最良構成) と同一のフラグ集合。production_config.advantage_overlay_
    flags() は resolved-exchange-eval 系がまだ未登録のため使わない
    (docstring「土台」節参照)。
    """
    return dict(
        sample_interval=0.0,
        show_recognition=False,
        enable_early_fire_reaction=True,
        enable_per_side_settled=True,
        disable_score_lead_bias=True,
        disable_pressure=True,
        enable_counter_remaining_time=True,
        enable_counter_defender_only=True,
        stable_majority_window=True,
        enable_ojama_fall_placement_override=True,
        enable_ojama_fall_entry_hardening=True,
        enable_ojama_fall_scoped_exit=True,
        enable_resolved_exchange_eval=True,
        enable_resolved_decisive_amplify=True,
        enable_resolved_live_defender=True,
        enable_pseudo_chain_score_fill=True,
        layout="panel",
        render=False,
    )


def config_kwargs(tag: str) -> dict[str, bool]:
    """構成タグ (off/on) から指摘14 2フラグの kwargs を返す。"""
    is_on = tag == CONFIG_ON
    return dict(
        enable_resolved_live_defender_strict=is_on,
        enable_resolved_kill_override=is_on,
    )


def probe_duration_sec(path: Path) -> float:
    """動画の全長 [秒] を返す (ヘッダ読取のみ)。"""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n / fps if fps > 0 else 0.0


def video_id_of(filename: str) -> str:
    """ファイル名から video_id (例: "c10") を取り出す。"""
    return filename.removesuffix(".mp4").removeprefix("video_")


@dataclass(frozen=True)
class BacktestJob:
    """1 (動画, 地点, 構成) 分の収集ジョブ。"""

    video_id: str
    video_path: Path
    chunk_idx: int
    start_sec: float
    config_tag: str
    out_npz: Path
    log_path: Path


def build_jobs() -> list[BacktestJob]:
    """動画 × 3地点 × 2構成 のジョブ一覧を組み立てる。"""
    jobs: list[BacktestJob] = []
    for filename in VIDEO_FILENAMES:
        vid = video_id_of(filename)
        video_path = VIDEO_DIR / filename
        if not video_path.exists():
            print(f"[skip] 動画が無い: {video_path}")
            continue
        duration = probe_duration_sec(video_path)
        if duration <= 0:
            print(f"[skip] 動画を開けない: {video_path}")
            continue
        for k, frac in enumerate(CHUNK_OFFSET_FRACTIONS[:N_CHUNKS_PER_VIDEO]):
            start_sec = max(0.0, frac * duration)
            for tag in (CONFIG_OFF, CONFIG_ON):
                out_dir = OUT_ROOT / "npz" / tag
                log_dir = LOG_ROOT / tag
                out_dir.mkdir(parents=True, exist_ok=True)
                log_dir.mkdir(parents=True, exist_ok=True)
                name = f"{vid}_chunk{k}"
                jobs.append(BacktestJob(
                    video_id=vid, video_path=video_path, chunk_idx=k,
                    start_sec=start_sec, config_tag=tag,
                    out_npz=out_dir / f"{name}.npz",
                    log_path=log_dir / f"{name}.log",
                ))
    return jobs


# BLAS/OMP のスレッド過剰発行を防ぐ環境変数 (2026-08-15 是正)。
# 初回実測: 10並列で20分経過しても1ジョブも完了せず (単独実行時の推定より
# 5倍以上遅い)。原因=各プロセスが numpy/torch/opencv のBLAS並列化で複数
# スレッドを立て、10並列 x 数スレッド = 16コアに対し大幅オーバーサブスクリプション
# していたため (feedback_subprocess_env_propagation: env は {**os.environ,...}
# で完全上書きしない形式必須)。scripts/_gen_demo_final3_2026-08-15.sh が
# 実写生成時に同じ変数を export しているのと同じ対処。
_SINGLE_THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def run_job(job: BacktestJob) -> tuple[str, bool, float]:
    """1ジョブを subprocess (--worker モード) で実行する。"""
    job_name = f"{job.config_tag}/{job.video_id}_chunk{job.chunk_idx}"
    if job.out_npz.exists():
        # 再実行時の重複計算を避ける (中断からの再開に対応、
        # feedback「長時間ジョブはチェックポイントを書きながら」)。
        return job_name, True, 0.0
    cmd = [
        str(PYTHON_BIN), "-u", "-m",
        "scripts._backtest_issue14_flags_2026-08-15",
        "--worker",
        "--video", str(job.video_path),
        "--config", job.config_tag,
        "--start-sec", str(job.start_sec),
        "--chunk-sec", str(CHUNK_SEC),
        "--out-npz", str(job.out_npz),
    ]
    start = time.monotonic()
    env = {**os.environ, **_SINGLE_THREAD_ENV}
    with job.log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    elapsed = time.monotonic() - start
    ok = proc.returncode == 0 and job.out_npz.exists()
    print(f"[{'OK' if ok else 'FAIL'}] {job_name} ({elapsed:.1f}s)")
    return job_name, ok, elapsed


def run_driver() -> None:
    """全ジョブを並列実行するドライバ。完了後に集計まで行う。"""
    jobs = build_jobs()
    n_videos = len({j.video_id for j in jobs})
    print(
        f"[driver] ジョブ数: {len(jobs)} (動画{n_videos}本 × "
        f"最大{N_CHUNKS_PER_VIDEO}地点 × 2構成、CHUNK_SEC={CHUNK_SEC})"
    )
    t0 = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as ex:
        for r in ex.map(run_job, jobs):
            results.append(r)
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"[driver] 完了: {n_ok}/{len(results)} 成功 ({time.monotonic() - t0:.1f}s)")
    if n_ok < len(results):
        print("[driver] 失敗ジョブ:")
        for name, ok, _ in results:
            if not ok:
                print(f"  {name}")
    run_aggregate()


# ============================
# --worker モード: 1 (動画,地点,構成) を実処理する
# ============================

def _run_worker(
    video: Path, config_tag: str, start_sec: float, chunk_sec: float, out_npz: Path,
) -> None:
    """1ジョブを実処理する (決着ホールド活性トレース計装込み)。

    generate() 自体は無変更のまま呼び出し、`ResolvedExchangeTracker.update`
    だけを read-only ラッパーに差し替える (モジュール属性の差し替えはこの
    プロセス内のみに閉じる、1ジョブ限りで終了するため後始末は不要)。
    """
    import scripts.visualize_advantage_overlay as ov

    hold_trace: list[tuple[float, bool]] = []
    original_update = ov.ResolvedExchangeTracker.update

    def _traced_update(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        active, just_deactivated = original_update(self, *args, **kwargs)
        # 呼出側 (generate() 内、update(r_p1, r_p2, snap, elapsed_sec, t_sec=t,
        # b1=b1, b2=b2) の実際の位置引数は (r_p1, r_p2, snap, elapsed_sec)。
        # t_sec は常に kwarg 渡しのため kwargs 優先、位置引数のみの旧呼出
        # 互換のため args[3]=elapsed_sec へフォールバックする。
        t_sec = kwargs.get("t_sec")
        if t_sec is None and len(args) >= 4:
            t_sec = args[3]  # elapsed_sec フォールバック (t_sec省略時、旧呼出互換)
        hold_trace.append((float(t_sec) if t_sec is not None else float("nan"), bool(active)))
        return active, just_deactivated

    ov.ResolvedExchangeTracker.update = _traced_update
    history: list[tuple[float, float]] = []
    try:
        kwargs = dict(common_overlay_kwargs())
        kwargs.update(config_kwargs(config_tag))
        ov.generate(
            video, Path("/tmp/_unused_issue14_backtest.mp4"),
            max_sec=0.0, start_sec=start_sec, end_sec=start_sec + chunk_sec,
            debug_history_out=history,
            **kwargs,
        )
    finally:
        ov.ResolvedExchangeTracker.update = original_update

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    t_adv = np.array([t for t, _ in history], dtype=np.float64)
    adv = np.array([a for _, a in history], dtype=np.float32)
    t_hold = np.array([t for t, _ in hold_trace], dtype=np.float64)
    active = np.array([a for _, a in hold_trace], dtype=np.bool_)
    np.savez_compressed(
        out_npz, t_adv=t_adv, adv=adv, t_hold=t_hold, active=active,
        start_sec=np.float64(start_sec), chunk_sec=np.float64(chunk_sec),
    )
    print(f"[worker] adv={len(history)} hold={len(hold_trace)} -> {out_npz}")


# ============================
# 集計
# ============================

def _hold_events(t_hold: np.ndarray, active: np.ndarray) -> list[tuple[float, float]]:
    """活性トレースから決着ホールド1回分ごとの (開始t, 継続秒) 一覧を作る。"""
    events: list[tuple[float, float]] = []
    start_t: float | None = None
    prev_t: float | None = None
    for t, is_active in zip(t_hold, active):
        if is_active and start_t is None:
            start_t = float(t)
        if not is_active and start_t is not None:
            events.append((start_t, float(prev_t) - start_t))
            start_t = None
        prev_t = float(t)
    if start_t is not None and prev_t is not None:
        events.append((start_t, float(prev_t) - start_t))
    return events


def _frozen_runs(t_adv: np.ndarray, adv: np.ndarray) -> list[float]:
    """disp_adv が連続して不変だった区間の継続秒リストを返す。"""
    if len(t_adv) < 2:
        return []
    runs: list[float] = []
    run_start = t_adv[0]
    for i in range(1, len(adv)):
        if adv[i] != adv[i - 1]:
            runs.append(float(t_adv[i - 1] - run_start))
            run_start = t_adv[i]
    runs.append(float(t_adv[-1] - run_start))
    return runs


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.array(values), q)) if values else float("nan")


# 上位乖離場面のうち実画面フレームを抽出する件数
# (feedback_review_actual_screen_frames、抽象プロットでなく実画面で判断材料を残す)
N_EXTRACT_FRAMES: int = 8
# 同一 (video_chunk) 内で近接時刻を重複選出しない最小間隔 [秒]
# (同一決着ホールド内の隣接フレームばかり選ぶのを防ぎ、場面の多様性を確保する)
DIVERSIFY_MIN_GAP_SEC: float = 3.0


def _diversified_top(
    top_diffs: list[tuple[float, str, float, float, float]], n: int,
) -> list[tuple[float, str, float, float, float]]:
    """|diff| 降順のリストから、同一場面の重複を避けつつ上位 n 件を選ぶ。"""
    selected: list[tuple[float, str, float, float, float]] = []
    last_t_by_name: dict[str, list[float]] = {}
    for row in top_diffs:
        _, name, t, _, _ = row
        prev_ts = last_t_by_name.get(name, [])
        if any(abs(t - pt) < DIVERSIFY_MIN_GAP_SEC for pt in prev_ts):
            continue
        selected.append(row)
        last_t_by_name.setdefault(name, []).append(t)
        if len(selected) >= n:
            break
    return selected


def _video_path_for(name: str) -> "Path | None":
    """"{video_id}_chunk{k}" 形式の name から元動画パスを復元する。"""
    vid = name.split("_chunk")[0]
    path = VIDEO_DIR / f"video_{vid}.mp4"
    return path if path.exists() else None


def extract_top_frames(top_diffs: list[tuple[float, str, float, float, float]]) -> list[dict]:
    """上位乖離場面から実画面フレームを抽出し PNG 保存する (read-only)。

    ネイティブ解像度のまま1フレームだけ cv2 で取り出す (認識・判定は一切
    再実行しない、証拠画像としての単純な frame dump)。
    """
    frames_dir = OUT_ROOT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict] = []
    seen_video: dict[str, "cv2.VideoCapture"] = {}
    for d, name, t, off_adv, on_adv in _diversified_top(top_diffs, N_EXTRACT_FRAMES):
        video_path = _video_path_for(name)
        if video_path is None:
            continue
        key = str(video_path)
        if key not in seen_video:
            seen_video[key] = cv2.VideoCapture(key)
        cap = seen_video[key]
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS_FALLBACK
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[frame-extract skip] 読めない: {name} t={t:.2f}")
            continue
        out_path = frames_dir / f"{name}_t{t:.2f}_diff{d:.0f}.png"
        cv2.imwrite(str(out_path), frame)
        extracted.append(dict(name=name, t=t, off_adv=off_adv, on_adv=on_adv,
                              diff=d, path=str(out_path)))
        print(f"[frame-extract] {out_path}")
    for cap in seen_video.values():
        cap.release()
    return extracted


def run_aggregate() -> None:
    """収集済み npz を集計し summary.md/json + 上位乖離フレームを書き出す。"""
    from scripts.visualize_advantage_overlay import adv_to_winprob

    npz_root = OUT_ROOT / "npz"
    off_files = sorted((npz_root / CONFIG_OFF).glob("*.npz"))
    pairs = []
    for off_path in off_files:
        on_path = npz_root / CONFIG_ON / off_path.name
        if not on_path.exists():
            print(f"[aggregate skip] ON側が無い: {on_path}")
            continue
        pairs.append((off_path.stem, off_path, on_path))

    all_diffs: list[float] = []
    changed_frames = 0
    total_matched_frames = 0
    extreme_frac = {CONFIG_OFF: [], CONFIG_ON: []}
    frozen_runs_all = {CONFIG_OFF: [], CONFIG_ON: []}
    hold_events_all = {CONFIG_OFF: [], CONFIG_ON: []}
    top_diffs: list[tuple[float, str, float, float, float]] = []  # (|diff|, name, t, off_adv, on_adv)
    per_pair_rows: list[dict] = []

    for name, off_path, on_path in pairs:
        off_d = np.load(off_path)
        on_d = np.load(on_path)
        t_off, adv_off = off_d["t_adv"], off_d["adv"]
        t_on, adv_on = on_d["t_adv"], on_d["adv"]
        off_map = {round(float(t), T_MATCH_DECIMALS): float(a) for t, a in zip(t_off, adv_off)}
        on_map = {round(float(t), T_MATCH_DECIMALS): float(a) for t, a in zip(t_on, adv_on)}
        common_t = sorted(set(off_map) & set(on_map))
        n_pair_changed = 0
        for t in common_t:
            a_off, a_on = off_map[t], on_map[t]
            diff = a_on - a_off
            total_matched_frames += 1
            if abs(diff) > 1e-6:
                changed_frames += 1
                n_pair_changed += 1
                all_diffs.append(abs(diff))
                top_diffs.append((abs(diff), name, t, a_off, a_on))
        p1_off = np.array([adv_to_winprob(a) for a in adv_off]) if len(adv_off) else np.array([])
        p1_on = np.array([adv_to_winprob(a) for a in adv_on]) if len(adv_on) else np.array([])
        if len(p1_off):
            extreme_frac[CONFIG_OFF].append(
                float(np.mean((p1_off > EXTREME_P1_HIGH) | (p1_off < EXTREME_P1_LOW))))
        if len(p1_on):
            extreme_frac[CONFIG_ON].append(
                float(np.mean((p1_on > EXTREME_P1_HIGH) | (p1_on < EXTREME_P1_LOW))))
        frozen_runs_all[CONFIG_OFF].extend(_frozen_runs(t_off, adv_off))
        frozen_runs_all[CONFIG_ON].extend(_frozen_runs(t_on, adv_on))
        hold_events_all[CONFIG_OFF].extend(_hold_events(off_d["t_hold"], off_d["active"]))
        hold_events_all[CONFIG_ON].extend(_hold_events(on_d["t_hold"], on_d["active"]))
        per_pair_rows.append(dict(
            name=name, n_common=len(common_t), n_changed=n_pair_changed,
            n_hold_off=len(_hold_events(off_d["t_hold"], off_d["active"])),
            n_hold_on=len(_hold_events(on_d["t_hold"], on_d["active"])),
        ))

    top_diffs.sort(key=lambda r: -r[0])
    summary = dict(
        n_pairs=len(pairs),
        total_matched_frames=total_matched_frames,
        changed_frames=changed_frames,
        changed_fraction=(changed_frames / total_matched_frames) if total_matched_frames else 0.0,
        diff_median=_percentile(all_diffs, 50),
        diff_p95=_percentile(all_diffs, 95),
        diff_max=(max(all_diffs) if all_diffs else 0.0),
        extreme_p1_fraction_off=float(np.mean(extreme_frac[CONFIG_OFF])) if extreme_frac[CONFIG_OFF] else 0.0,
        extreme_p1_fraction_on=float(np.mean(extreme_frac[CONFIG_ON])) if extreme_frac[CONFIG_ON] else 0.0,
        frozen_run_median_off=_percentile(frozen_runs_all[CONFIG_OFF], 50),
        frozen_run_p95_off=_percentile(frozen_runs_all[CONFIG_OFF], 95),
        frozen_run_max_off=(max(frozen_runs_all[CONFIG_OFF]) if frozen_runs_all[CONFIG_OFF] else 0.0),
        frozen_run_median_on=_percentile(frozen_runs_all[CONFIG_ON], 50),
        frozen_run_p95_on=_percentile(frozen_runs_all[CONFIG_ON], 95),
        frozen_run_max_on=(max(frozen_runs_all[CONFIG_ON]) if frozen_runs_all[CONFIG_ON] else 0.0),
        n_hold_events_off=len(hold_events_all[CONFIG_OFF]),
        n_hold_events_on=len(hold_events_all[CONFIG_ON]),
        hold_duration_median_off=_percentile([d for _, d in hold_events_all[CONFIG_OFF]], 50),
        hold_duration_median_on=_percentile([d for _, d in hold_events_all[CONFIG_ON]], 50),
        hold_duration_total_off=sum(d for _, d in hold_events_all[CONFIG_OFF]),
        hold_duration_total_on=sum(d for _, d in hold_events_all[CONFIG_ON]),
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    lines = ["# 指摘14 2フラグ 全域バックテスト集計 (2026-08-15)", ""]
    lines.append(f"- 動画/地点ペア数: {summary['n_pairs']}")
    lines.append(f"- マッチしたフレーム総数: {summary['total_matched_frames']}")
    lines.append(
        f"- 値が変化したフレーム数/割合: {summary['changed_frames']} "
        f"({summary['changed_fraction']*100:.3f}%)")
    lines.append(
        f"- 変化量(|ON-OFF|) 中央値/p95/最大: "
        f"{summary['diff_median']:.2f} / {summary['diff_p95']:.2f} / {summary['diff_max']:.2f}")
    lines.append(
        f"- 極端値(p1>95% or <5%)時間割合: OFF={summary['extreme_p1_fraction_off']*100:.2f}% "
        f"-> ON={summary['extreme_p1_fraction_on']*100:.2f}%")
    lines.append(
        f"- 凍結継続時間 中央値/p95/最大 [秒]: OFF="
        f"{summary['frozen_run_median_off']:.3f}/{summary['frozen_run_p95_off']:.3f}/"
        f"{summary['frozen_run_max_off']:.3f} -> ON="
        f"{summary['frozen_run_median_on']:.3f}/{summary['frozen_run_p95_on']:.3f}/"
        f"{summary['frozen_run_max_on']:.3f}")
    lines.append(
        f"- 決着ホールド発生回数: OFF={summary['n_hold_events_off']} -> ON={summary['n_hold_events_on']}")
    lines.append(
        f"- 決着ホールド継続時間 中央値/合計 [秒]: OFF="
        f"{summary['hold_duration_median_off']:.3f}/{summary['hold_duration_total_off']:.1f} -> ON="
        f"{summary['hold_duration_median_on']:.3f}/{summary['hold_duration_total_on']:.1f}")
    lines.append("")
    lines.append("## 上位乖離場面 (|ON-OFF| 降順、上位15件)")
    lines.append("| video_chunk | t_abs[s] | OFF adv | ON adv | diff |")
    lines.append("|---|---|---|---|---|")
    for d, n, t, o, k in top_diffs[:15]:
        lines.append(f"| {n} | {t:.2f} | {o:+.1f} | {k:+.1f} | {d:.1f} |")

    extracted = extract_top_frames(top_diffs)
    lines.append("")
    lines.append(f"## 実画面フレーム抽出 ({len(extracted)}件、上位乖離場面から)")
    for row in extracted:
        lines.append(
            f"- {row['name']} t={row['t']:.2f}s diff={row['diff']:.1f} "
            f"(OFF={row['off_adv']:+.1f}/ON={row['on_adv']:+.1f}): {row['path']}")

    (OUT_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with (OUT_ROOT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, per_pair=per_pair_rows,
                       top_diffs=[dict(abs_diff=d, name=n, t=t, off_adv=o, on_adv=k)
                                  for d, n, t, o, k in top_diffs[:30]],
                       extracted_frames=extracted),
                  f, ensure_ascii=False, indent=2)
    print(f"[aggregate] summary -> {OUT_ROOT / 'summary.md'}")
    print(f"[aggregate] top diffs -> {len(top_diffs)} 件中上位15件を記録、"
          f"実画面フレーム{len(extracted)}件抽出")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--config", choices=[CONFIG_OFF, CONFIG_ON])
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--chunk-sec", type=float, default=0.0)
    ap.add_argument("--out-npz", type=Path)
    args = ap.parse_args()

    if args.worker:
        assert args.video and args.config and args.out_npz
        _run_worker(args.video, args.config, args.start_sec, args.chunk_sec, args.out_npz)
        return 0

    if args.aggregate_only:
        run_aggregate()
        return 0

    run_driver()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
