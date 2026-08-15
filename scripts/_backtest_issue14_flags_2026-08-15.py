"""指摘14修正2フラグ (--resolved-live-defender-strict / --resolved-kill-override)
の全域バックテスト (2026-08-15、coordinator依頼)。

## v2 改定 (2026-08-15、同日中の是正・2点のみ)
初回実行 (data/verify/backtest_issue14_2026-08-15/) に2つの問題が判明した
ための再実行。**旧結果は上書きしない** (OUT_ROOT/LOG_ROOT を v2 パスへ変更)。

1. **フラグ構成が出荷構成と不一致**: `common_overlay_kwargs()` が
   `stable_majority_window` / `enable_ojama_fall_entry_hardening` /
   `enable_ojama_fall_scoped_exit` を True にしていたが、この3つは
   2026-08-15 に不採用確定 (効果ゼロ/悪化主犯/寄与ゼロ)。
   `scripts/_gen_demo_final4_2026-08-15.sh` (出荷直前構成) と完全一致させる
   よう削除した。
2. **凍結検出器 (`_frozen_runs`) が試合外画面を凍結として誤計上** (弱点台帳
   W14)。`RecognitionPipeline.update` を read-only ラッパーで計装し、
   disp_adv と同一フレーム粒度で state1/state2 (BoardState.name) と
   score1/score2 を記録。凍結区間の判定に「試合外
   (state1==MENU or state2==MENU or 両者スコアが SCORE_NEAR_ZERO_THRESHOLD
   以下)」フィルタを追加し、該当区間は「凍結」統計から除外する。
   さらに user 指定の判定基準 (★最重要) に基づき、試合中の凍結区間を
   A (全時間帯 両者連鎖中=正当) / B (途中から片側が設置開始したのに
   固定=不備候補) / C (その他) へ分類する。


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
# v2 (2026-08-15): 旧結果 (data/verify/backtest_issue14_2026-08-15/) を
# 上書きしないよう別ディレクトリに出力する。
OUT_ROOT: Path = _ROOT / "data" / "verify" / "backtest_issue14_v2_2026-08-15"
LOG_ROOT: Path = _ROOT / "logs" / "backtest_issue14_v2_2026-08-15"
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
# 同時実行プロセス数 (v2: coordinator指示により最大8に変更)
MAX_PARALLEL_WORKERS: int = 8

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

# ============================
# v2 是正 (問題2): 試合外フィルタ + A/B/C 分類用定数
# ============================

# BoardState.name (src/board_state_machine.py) のうち「連鎖中」とみなす値。
# ★user判定基準: この2値のまま両者とも動かないなら凍結は正当。
CHAIN_LIKE_STATE_NAMES: frozenset[str] = frozenset({"CHAIN", "GRAVITY_SETTLE"})
# 「設置活動中/確定済」とみなす値。連鎖後にこれへ遷移したのに disp_adv が
# 固定されたままなら不備の候補 (B)。
PLACE_LIKE_STATE_NAMES: frozenset[str] = frozenset({"STABLE", "TSUMO_FALL"})
# 試合外 (タイトル/リザルト/リトライ) を表す BoardState.name。
MENU_STATE_NAME: str = "MENU"
# score が OCR 失敗 (None) だった場合の npz 保存用センチネル値
# (実スコアは非負のため -1 は衝突しない)。
SCORE_NONE_SENTINEL: int = -1
# 凍結区間分類: A=正当(両者連鎖中) / B=不備候補(片側設置開始なのに固定) /
# C=その他 (盤面が実際に不変等)。
RUN_CLASS_LEGIT: str = "A"
RUN_CLASS_SUSPECT: str = "B"
RUN_CLASS_OTHER: str = "C"


def common_overlay_kwargs() -> dict[str, object]:
    """OFF/ON 双方に共通の土台 kwargs を返す。

    scripts/_gen_demo_final4_2026-08-15.sh (2026-08-15 出荷直前構成、
    指摘14修正2フラグを除く全フラグ) と完全一致させる。
    [v2是正] stable_majority_window / enable_ojama_fall_entry_hardening /
    enable_ojama_fall_scoped_exit の3つは 2026-08-15 に不採用確定
    (効果ゼロ/悪化主犯/寄与ゼロ、final4 スクリプトのコメント参照) のため
    削除した (旧v1では誤って True のまま計測していた)。
    enable_ojama_fall_placement_override は採用済みのため維持。
    production_config.advantage_overlay_flags() は resolved-exchange-eval
    系がまだ未登録のため使わない (docstring「土台」節参照)。
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
        enable_ojama_fall_placement_override=True,
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
    from src.recognition_pipeline import RecognitionPipeline

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

    # v2 追加 (問題2): RecognitionPipeline.update を read-only ラッパーへ
    # 差し替え、disp_adv と同一フレーム粒度 (generate() 内 t = fi/fps を
    # そのまま pipe.update に渡している、line "r = pipe.update(fi, t,
    # recog_frame)") で state1/state2/score1/score2 を記録する。
    # production コード自体は無変更 (このプロセス内限定の属性差し替え、
    # ResolvedExchangeTracker.update の差し替えと同じ方式)。
    state_trace: list[tuple[float, str, str, int, int]] = []
    original_pipe_update = RecognitionPipeline.update

    def _traced_pipe_update(self, frame_idx, time_sec, frame):  # noqa: ANN001
        result = original_pipe_update(self, frame_idx, time_sec, frame)
        s1 = getattr(result.p1.state, "name", "")
        s2 = getattr(result.p2.state, "name", "")
        sc1 = result.p1.score if result.p1.score is not None else SCORE_NONE_SENTINEL
        sc2 = result.p2.score if result.p2.score is not None else SCORE_NONE_SENTINEL
        state_trace.append((float(time_sec), s1, s2, int(sc1), int(sc2)))
        return result

    ov.ResolvedExchangeTracker.update = _traced_update
    RecognitionPipeline.update = _traced_pipe_update
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
        RecognitionPipeline.update = original_pipe_update

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    t_adv = np.array([t for t, _ in history], dtype=np.float64)
    adv = np.array([a for _, a in history], dtype=np.float32)
    t_hold = np.array([t for t, _ in hold_trace], dtype=np.float64)
    active = np.array([a for _, a in hold_trace], dtype=np.bool_)
    t_state = np.array([t for t, _, _, _, _ in state_trace], dtype=np.float64)
    state1 = np.array([s1 for _, s1, _, _, _ in state_trace], dtype="U20")
    state2 = np.array([s2 for _, _, s2, _, _ in state_trace], dtype="U20")
    score1 = np.array([sc1 for _, _, _, sc1, _ in state_trace], dtype=np.int64)
    score2 = np.array([sc2 for _, _, _, _, sc2 in state_trace], dtype=np.int64)
    np.savez_compressed(
        out_npz, t_adv=t_adv, adv=adv, t_hold=t_hold, active=active,
        t_state=t_state, state1=state1, state2=state2,
        score1=score1, score2=score2,
        start_sec=np.float64(start_sec), chunk_sec=np.float64(chunk_sec),
    )
    print(f"[worker] adv={len(history)} hold={len(hold_trace)} "
          f"state={len(state_trace)} -> {out_npz}")


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


def _frozen_run_windows(
    t_adv: np.ndarray, adv: np.ndarray,
) -> list[tuple[float, float, float]]:
    """disp_adv が連続して不変だった区間の (開始t, 終了t, 継続秒) 一覧を返す。

    v2: 単なる継続秒だけでなく開始/終了時刻も返す (問題2是正、試合外フィルタ
    /A-B-C分類のために区間内の state トレースへ後で突合わせる必要があるため)。

    [v2実行中に発覚した測定器バグの是正・第3の修正] disp_adv が毎フレーム
    変化し続ける (=凍結が一切無い) 区間では、隣接2点が「異なる」ことをもって
    「1点だけの縮退run (継続秒=0)」が全フレームぶん記録されてしまい、
    (a) 凍結件数が実態の数十〜数百倍に水増しされる (b) A/B/C分類が単一
    フレームの状態タプルに対して評価され、たまたま隣接側が CHAIN で自側が
    STABLE/TSUMO_FALL のような何気ない瞬間を大量に「B(不備候補)」誤判定する
    という二重の実害を引き起こした (実測: video_c10 1本だけで270件超のB誤検出
    → フレーム抽出が暴走、ディスク圧迫の危険まで確認)。継続秒=0 は定義上
    「1フレームも重複していない」= そもそも凍結ではないため、ここで除外する
    (この不具合は v1 の生凍結分布集計にも内在していたため、v2 で合わせて
    是正する)。
    """
    if len(t_adv) < 2:
        return []
    windows: list[tuple[float, float, float]] = []
    run_start = t_adv[0]
    for i in range(1, len(adv)):
        if adv[i] != adv[i - 1]:
            windows.append((float(run_start), float(t_adv[i - 1]), float(t_adv[i - 1] - run_start)))
            run_start = t_adv[i]
    windows.append((float(run_start), float(t_adv[-1]), float(t_adv[-1] - run_start)))
    # 継続秒=0 (縮退run、1フレームも重複していない) を除外する。
    return [w for w in windows if w[2] > 0.0]


def _frozen_runs(t_adv: np.ndarray, adv: np.ndarray) -> list[float]:
    """disp_adv が連続して不変だった区間の継続秒リストを返す (旧仕様、生値)。

    v2 でも「是正前の生の凍結分布」を報告する (前回結果との差分を示すため)
    目的で残す。試合外フィルタは適用しない。
    """
    return [dur for _, _, dur in _frozen_run_windows(t_adv, adv)]


def _state_lookup(state_data: dict) -> dict[float, tuple[str, str, int, int]]:
    """npz の t_state/state1/state2/score1/score2 から t(丸め)→状態タプルの辞書を作る。

    generate() 内で t = fi/fps を pipe.update()/history 双方に同一の float
    として渡しているため (同一プロセス内の同一変数)、丸め誤差はほぼ無いが
    OFF/ON マージと同じ T_MATCH_DECIMALS 丸めを踏襲し安全側に倒す。
    """
    t_state = state_data["t_state"]
    s1 = state_data["state1"]
    s2 = state_data["state2"]
    sc1 = state_data["score1"]
    sc2 = state_data["score2"]
    return {
        round(float(t), T_MATCH_DECIMALS): (str(s1[i]), str(s2[i]), int(sc1[i]), int(sc2[i]))
        for i, t in enumerate(t_state)
    }


def _score_looks_waiting(score: int) -> bool:
    """スコアが OCR失敗(SCORE_NONE_SENTINEL)、または0付近(待機画面)かを返す。"""
    from scripts.visualize_advantage_overlay import SCORE_NEAR_ZERO_THRESHOLD
    return score == SCORE_NONE_SENTINEL or score <= SCORE_NEAR_ZERO_THRESHOLD


def _states_in_window(
    lookup: dict[float, tuple[str, str, int, int]], start_t: float, end_t: float,
) -> list[tuple[str, str, int, int]]:
    """凍結区間 [start_t, end_t] (両端含む) 内の state トレース点を時刻昇順で返す。"""
    lo = round(start_t, T_MATCH_DECIMALS)
    hi = round(end_t, T_MATCH_DECIMALS)
    return [v for t, v in sorted(lookup.items()) if lo <= t <= hi]


def _is_out_of_match_window(states: list[tuple[str, str, int, int]]) -> bool:
    """区間内に試合外 (MENU) または両者スコア待機状態のフレームが1つでもあれば True。

    v2是正 (問題2、弱点台帳W14): 「3.63秒 ちょうど50.0%固定」等の凍結誤計上
    (実体はラウンド間待機/ばたんきゅー演出) をここで除外する。
    state が全く取れない (states空、= state トレースと disp_adv のt突合わせに
    失敗) 場合は判定不能として False (=試合中として扱う、安全側フォールバック
    だが件数を "unmatched" として別途カウントし正直に開示する)。
    """
    for s1, s2, sc1, sc2 in states:
        if s1 == MENU_STATE_NAME or s2 == MENU_STATE_NAME:
            return True
        if _score_looks_waiting(sc1) and _score_looks_waiting(sc2):
            return True
    return False


def _classify_run(states: list[tuple[str, str, int, int]]) -> str:
    """★user判定基準に基づき凍結区間を A(正当)/B(不備候補)/C(その他) に分類する。

    A: 全時間帯で両者とも連鎖中 (CHAIN_LIKE_STATE_NAMES) のまま。
    B: 区間内のどこかで連鎖中(いずれかの側)から設置活動中/確定済
       (PLACE_LIKE_STATE_NAMES) へ遷移した形跡があるのに disp_adv は不変
       (= 置くたびに微妙に変動するはず、というuser基準への違反候補)。
    C: そのいずれでもない (連鎖自体が一度も観測されない=盤面が実際に不変等)。
    """
    has_chain_like = any(
        s1 in CHAIN_LIKE_STATE_NAMES or s2 in CHAIN_LIKE_STATE_NAMES for s1, s2, _, _ in states
    )
    if not has_chain_like:
        return RUN_CLASS_OTHER
    all_chain_like = all(
        s1 in CHAIN_LIKE_STATE_NAMES and s2 in CHAIN_LIKE_STATE_NAMES for s1, s2, _, _ in states
    )
    if all_chain_like:
        return RUN_CLASS_LEGIT
    # 連鎖中フレームが少なくとも1つあり、かつ全時間帯連鎖中ではない
    # (= どこかで片側が PLACE_LIKE へ遷移している) → 不備候補。
    has_place_like = any(
        s1 in PLACE_LIKE_STATE_NAMES or s2 in PLACE_LIKE_STATE_NAMES for s1, s2, _, _ in states
    )
    if has_place_like:
        return RUN_CLASS_SUSPECT
    return RUN_CLASS_OTHER


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


# B(不備候補) 実画面フレーム抽出の上限件数 (指示上「数件なら」保存する想定の
# 安全弁。件数がこれを超える場合はディスク圧迫防止のため上位のみ抽出し、
# 打ち切ったことを summary に明記する)。
MAX_B_EVENT_FRAMES: int = 20
# B(不備候補) 全件リストの markdown 表示上限 (JSON には常に全件を残す)。
B_EVENT_MD_TABLE_CAP: int = 300


def extract_b_event_frames(b_events: list[dict]) -> list[dict]:
    """B分類 (不備候補) の凍結区間から、開始時刻の実画面フレームを保存する。

    件数が少数 (指示上「数件なら」) を想定した単純実装。MAX_B_EVENT_FRAMES
    件を上限に保存する (feedback_review_actual_screen_frames、抽象プロット
    でなく実画面で判断。無制限だとディスク圧迫の危険があるため安全弁を設ける)。
    継続時間が長い順に優先する (=より深刻な不備候補を優先的に証拠化)。
    """
    if not b_events:
        return []
    b_events = sorted(b_events, key=lambda e: -e["duration"])[:MAX_B_EVENT_FRAMES]
    frames_dir = OUT_ROOT / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict] = []
    seen_video: dict[str, "cv2.VideoCapture"] = {}
    for ev in b_events:
        video_path = _video_path_for(ev["name"])
        if video_path is None:
            continue
        key = str(video_path)
        if key not in seen_video:
            seen_video[key] = cv2.VideoCapture(key)
        cap = seen_video[key]
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS_FALLBACK
        t = ev["start_t"]
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[B-frame-extract skip] 読めない: {ev['name']} t={t:.2f}")
            continue
        out_path = frames_dir / (
            f"Bevent_{ev['config']}_{ev['name']}_t{t:.2f}_dur{ev['duration']:.2f}.png")
        cv2.imwrite(str(out_path), frame)
        row = dict(ev)
        row["path"] = str(out_path)
        extracted.append(row)
        print(f"[B-frame-extract] {out_path}")
    for cap in seen_video.values():
        cap.release()
    return extracted


def run_aggregate() -> None:
    """収集済み npz を集計し summary.md/json + 上位乖離/B事象フレームを書き出す。

    v2是正 (問題2): 凍結区間を試合外フィルタ (MENU state / 待機スコア) で
    除外し、試合中の凍結区間のみを A(正当)/B(不備候補)/C(その他) に分類する
    (★user判定基準)。旧v1相当の「生の凍結分布」も差分報告用に残す。
    """
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
    # 生の凍結分布 (試合外フィルタ適用前、旧v1相当。差分報告用に残す)。
    frozen_runs_raw = {CONFIG_OFF: [], CONFIG_ON: []}
    # 試合外フィルタ適用後 (v2是正済み) の凍結分布。
    frozen_runs_inmatch = {CONFIG_OFF: [], CONFIG_ON: []}
    n_runs_excluded_out_of_match = {CONFIG_OFF: 0, CONFIG_ON: 0}
    # state トレースが空/突合わせ失敗で判定不能だった件数 (正直な開示用)。
    n_runs_unmatched_state = {CONFIG_OFF: 0, CONFIG_ON: 0}
    run_class_counts = {
        CONFIG_OFF: {RUN_CLASS_LEGIT: 0, RUN_CLASS_SUSPECT: 0, RUN_CLASS_OTHER: 0},
        CONFIG_ON: {RUN_CLASS_LEGIT: 0, RUN_CLASS_SUSPECT: 0, RUN_CLASS_OTHER: 0},
    }
    run_class_duration = {
        CONFIG_OFF: {RUN_CLASS_LEGIT: 0.0, RUN_CLASS_SUSPECT: 0.0, RUN_CLASS_OTHER: 0.0},
        CONFIG_ON: {RUN_CLASS_LEGIT: 0.0, RUN_CLASS_SUSPECT: 0.0, RUN_CLASS_OTHER: 0.0},
    }
    b_events: dict[str, list[dict]] = {CONFIG_OFF: [], CONFIG_ON: []}
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

        # --- 凍結区間: 生分布 + 試合外フィルタ + A/B/C分類 (v2是正) ---
        for tag, t_arr, adv_arr, d in (
            (CONFIG_OFF, t_off, adv_off, off_d), (CONFIG_ON, t_on, adv_on, on_d),
        ):
            windows = _frozen_run_windows(t_arr, adv_arr)
            frozen_runs_raw[tag].extend(dur for _, _, dur in windows)
            has_state = "t_state" in d.files and len(d["t_state"]) > 0
            lookup = _state_lookup(d) if has_state else {}
            for start_t, end_t, dur in windows:
                if not has_state:
                    n_runs_unmatched_state[tag] += 1
                    # 判定不能=フィルタ無しでフォールバック計上 (安全側、開示のみ)。
                    frozen_runs_inmatch[tag].append(dur)
                    continue
                states = _states_in_window(lookup, start_t, end_t)
                if not states:
                    n_runs_unmatched_state[tag] += 1
                    frozen_runs_inmatch[tag].append(dur)
                    continue
                if _is_out_of_match_window(states):
                    n_runs_excluded_out_of_match[tag] += 1
                    continue
                frozen_runs_inmatch[tag].append(dur)
                cls = _classify_run(states)
                run_class_counts[tag][cls] += 1
                run_class_duration[tag][cls] += dur
                if cls == RUN_CLASS_SUSPECT:
                    idx = int(np.argmin(np.abs(t_arr - start_t)))
                    frozen_val = float(adv_arr[idx])
                    b_events[tag].append(dict(
                        name=name, config=tag, start_t=start_t, end_t=end_t,
                        duration=dur, frozen_adv=frozen_val,
                        frozen_p1=float(adv_to_winprob(frozen_val)),
                    ))

        hold_events_all[CONFIG_OFF].extend(_hold_events(off_d["t_hold"], off_d["active"]))
        hold_events_all[CONFIG_ON].extend(_hold_events(on_d["t_hold"], on_d["active"]))
        per_pair_rows.append(dict(
            name=name, n_common=len(common_t), n_changed=n_pair_changed,
            n_hold_off=len(_hold_events(off_d["t_hold"], off_d["active"])),
            n_hold_on=len(_hold_events(on_d["t_hold"], on_d["active"])),
        ))

    top_diffs.sort(key=lambda r: -r[0])
    for tag in (CONFIG_OFF, CONFIG_ON):
        b_events[tag].sort(key=lambda e: (e["name"], e["start_t"]))

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
        # 生 (試合外フィルタ適用前、旧v1相当)
        frozen_run_median_off_raw=_percentile(frozen_runs_raw[CONFIG_OFF], 50),
        frozen_run_p95_off_raw=_percentile(frozen_runs_raw[CONFIG_OFF], 95),
        frozen_run_max_off_raw=(max(frozen_runs_raw[CONFIG_OFF]) if frozen_runs_raw[CONFIG_OFF] else 0.0),
        frozen_run_median_on_raw=_percentile(frozen_runs_raw[CONFIG_ON], 50),
        frozen_run_p95_on_raw=_percentile(frozen_runs_raw[CONFIG_ON], 95),
        frozen_run_max_on_raw=(max(frozen_runs_raw[CONFIG_ON]) if frozen_runs_raw[CONFIG_ON] else 0.0),
        n_runs_raw_off=len(frozen_runs_raw[CONFIG_OFF]),
        n_runs_raw_on=len(frozen_runs_raw[CONFIG_ON]),
        # 試合外フィルタ適用後 (v2是正済み、本命の数字)
        frozen_run_median_off=_percentile(frozen_runs_inmatch[CONFIG_OFF], 50),
        frozen_run_p95_off=_percentile(frozen_runs_inmatch[CONFIG_OFF], 95),
        frozen_run_max_off=(max(frozen_runs_inmatch[CONFIG_OFF]) if frozen_runs_inmatch[CONFIG_OFF] else 0.0),
        frozen_run_median_on=_percentile(frozen_runs_inmatch[CONFIG_ON], 50),
        frozen_run_p95_on=_percentile(frozen_runs_inmatch[CONFIG_ON], 95),
        frozen_run_max_on=(max(frozen_runs_inmatch[CONFIG_ON]) if frozen_runs_inmatch[CONFIG_ON] else 0.0),
        n_runs_excluded_out_of_match_off=n_runs_excluded_out_of_match[CONFIG_OFF],
        n_runs_excluded_out_of_match_on=n_runs_excluded_out_of_match[CONFIG_ON],
        n_runs_unmatched_state_off=n_runs_unmatched_state[CONFIG_OFF],
        n_runs_unmatched_state_on=n_runs_unmatched_state[CONFIG_ON],
        # A/B/C 分類 (試合中の凍結区間のみ、★user判定基準)
        run_class_counts_off=run_class_counts[CONFIG_OFF],
        run_class_counts_on=run_class_counts[CONFIG_ON],
        run_class_duration_off=run_class_duration[CONFIG_OFF],
        run_class_duration_on=run_class_duration[CONFIG_ON],
        n_hold_events_off=len(hold_events_all[CONFIG_OFF]),
        n_hold_events_on=len(hold_events_all[CONFIG_ON]),
        hold_duration_median_off=_percentile([d for _, d in hold_events_all[CONFIG_OFF]], 50),
        hold_duration_median_on=_percentile([d for _, d in hold_events_all[CONFIG_ON]], 50),
        hold_duration_total_off=sum(d for _, d in hold_events_all[CONFIG_OFF]),
        hold_duration_total_on=sum(d for _, d in hold_events_all[CONFIG_ON]),
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    lines = ["# 指摘14 2フラグ 全域バックテスト集計 v2 (2026-08-15是正)", ""]
    lines.append(
        "v1 (data/verify/backtest_issue14_2026-08-15/) からの是正2点: "
        "(1) 出荷構成一致 (不採用3フラグ削除) (2) 凍結検出器の試合外フィルタ+A/B/C分類")
    lines.append("")
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
    lines.append("")
    lines.append("## 凍結継続時間 (問題1: 出荷構成是正 / 問題2: 試合外フィルタ)")
    lines.append(
        f"- [是正前=生・v1相当] 件数/中央値/p95/最大 [秒]: "
        f"OFF={summary['n_runs_raw_off']}件 "
        f"{summary['frozen_run_median_off_raw']:.3f}/{summary['frozen_run_p95_off_raw']:.3f}/"
        f"{summary['frozen_run_max_off_raw']:.3f} -> "
        f"ON={summary['n_runs_raw_on']}件 "
        f"{summary['frozen_run_median_on_raw']:.3f}/{summary['frozen_run_p95_on_raw']:.3f}/"
        f"{summary['frozen_run_max_on_raw']:.3f}")
    lines.append(
        f"- 試合外として除外した区間数: OFF={summary['n_runs_excluded_out_of_match_off']}件 "
        f"-> ON={summary['n_runs_excluded_out_of_match_on']}件 "
        f"(判定不能でフォールバック計上: OFF={summary['n_runs_unmatched_state_off']}件 "
        f"ON={summary['n_runs_unmatched_state_on']}件)")
    lines.append(
        f"- [是正後=試合中のみ] 件数/中央値/p95/最大 [秒]: "
        f"OFF={len(frozen_runs_inmatch[CONFIG_OFF])}件 "
        f"{summary['frozen_run_median_off']:.3f}/{summary['frozen_run_p95_off']:.3f}/"
        f"{summary['frozen_run_max_off']:.3f} -> "
        f"ON={len(frozen_runs_inmatch[CONFIG_ON])}件 "
        f"{summary['frozen_run_median_on']:.3f}/{summary['frozen_run_p95_on']:.3f}/"
        f"{summary['frozen_run_max_on']:.3f}")
    lines.append("")
    lines.append("## ★最重要: 凍結区間の A(正当)/B(不備候補)/C(その他) 分類 (試合中のみ)")
    for tag, label in ((CONFIG_OFF, "OFF"), (CONFIG_ON, "ON")):
        c = run_class_counts[tag]
        du = run_class_duration[tag]
        lines.append(
            f"- {label}: A(両者連鎖中)={c[RUN_CLASS_LEGIT]}件/{du[RUN_CLASS_LEGIT]:.1f}秒 "
            f"B(不備候補)={c[RUN_CLASS_SUSPECT]}件/{du[RUN_CLASS_SUSPECT]:.1f}秒 "
            f"C(その他)={c[RUN_CLASS_OTHER]}件/{du[RUN_CLASS_OTHER]:.1f}秒")
    lines.append("")
    lines.append("### B(不備候補) 全件 (ON構成、出荷構成での実挙動)")
    lines.append(f"合計 {len(b_events[CONFIG_ON])} 件。")
    if b_events[CONFIG_ON]:
        _on_sorted = sorted(b_events[CONFIG_ON], key=lambda e: -e["duration"])
        lines.append("| video_chunk | 開始t[s] | 終了t[s] | 継続[s] | 固定値(adv) | 固定値(p1%) |")
        lines.append("|---|---|---|---|---|---|")
        for ev in _on_sorted[:B_EVENT_MD_TABLE_CAP]:
            lines.append(
                f"| {ev['name']} | {ev['start_t']:.2f} | {ev['end_t']:.2f} | "
                f"{ev['duration']:.2f} | {ev['frozen_adv']:+.1f} | {ev['frozen_p1']*100:.1f}% |")
        if len(_on_sorted) > B_EVENT_MD_TABLE_CAP:
            lines.append(
                f"(継続時間降順、上位{B_EVENT_MD_TABLE_CAP}件のみ表示。"
                f"全{len(_on_sorted)}件は summary.json 参照)")
    else:
        lines.append("無し (ON構成の試合中凍結区間に B 分類は0件)。")
    lines.append("")
    lines.append("### 参考: OFF構成側の B(不備候補)")
    lines.append(f"合計 {len(b_events[CONFIG_OFF])} 件。")
    if b_events[CONFIG_OFF]:
        _off_sorted = sorted(b_events[CONFIG_OFF], key=lambda e: -e["duration"])
        lines.append("| video_chunk | 開始t[s] | 終了t[s] | 継続[s] | 固定値(adv) | 固定値(p1%) |")
        lines.append("|---|---|---|---|---|---|")
        for ev in _off_sorted[:B_EVENT_MD_TABLE_CAP]:
            lines.append(
                f"| {ev['name']} | {ev['start_t']:.2f} | {ev['end_t']:.2f} | "
                f"{ev['duration']:.2f} | {ev['frozen_adv']:+.1f} | {ev['frozen_p1']*100:.1f}% |")
        if len(_off_sorted) > B_EVENT_MD_TABLE_CAP:
            lines.append(
                f"(継続時間降順、上位{B_EVENT_MD_TABLE_CAP}件のみ表示。"
                f"全{len(_off_sorted)}件は summary.json 参照)")
    else:
        lines.append("無し (OFF構成の試合中凍結区間に B 分類は0件)。")
    lines.append("")
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

    b_extracted = extract_b_event_frames(b_events[CONFIG_ON])
    lines.append("")
    lines.append(f"## B(不備候補) 実画面フレーム抽出 ({len(b_extracted)}件、ON構成)")
    if len(b_events[CONFIG_ON]) > len(b_extracted):
        lines.append(
            f"(全{len(b_events[CONFIG_ON])}件中、継続時間が長い順に上位"
            f"{MAX_B_EVENT_FRAMES}件のみ抽出。全件一覧は上表参照)")
    for row in b_extracted:
        lines.append(
            f"- {row['name']} t={row['start_t']:.2f}s dur={row['duration']:.2f}s "
            f"固定値={row['frozen_p1']*100:.1f}%: {row['path']}")

    (OUT_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    with (OUT_ROOT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(dict(summary=summary, per_pair=per_pair_rows,
                       top_diffs=[dict(abs_diff=d, name=n, t=t, off_adv=o, on_adv=k)
                                  for d, n, t, o, k in top_diffs[:30]],
                       extracted_frames=extracted,
                       b_events_on=b_events[CONFIG_ON], b_events_off=b_events[CONFIG_OFF],
                       b_extracted_frames=b_extracted),
                  f, ensure_ascii=False, indent=2)
    print(f"[aggregate] summary -> {OUT_ROOT / 'summary.md'}")
    print(f"[aggregate] top diffs -> {len(top_diffs)} 件中上位15件を記録、"
          f"実画面フレーム{len(extracted)}件抽出")
    print(f"[aggregate] B(不備候補): ON={len(b_events[CONFIG_ON])}件 OFF={len(b_events[CONFIG_OFF])}件、"
          f"実画面フレーム{len(b_extracted)}件抽出")


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
