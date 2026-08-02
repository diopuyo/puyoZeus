"""Step6 (2026-08-01): exchange_labels CSV に修正シミュ特徴量を付与する。

scripts/label_exchange_outcome.py が生成した exchange_labels*.csv の各発火
イベント行に対し、scripts/measure_exchange_effectiveness.py の
estimate_expected_net_damage (Step5) を計算し、3列を追加した拡張CSVを
出力する:
    sim_k_hands              : 着弾までに相手が打てる手数 (K、Step0で+1修正済み)
    sim_expected_counter_ojama: 相手の期待反撃量 (お邪魔換算、raw)
    sim_damage_score          : 正味ダメージスコア (0〜1)

既存資産を再実装しない (盤面復元・被覆状態分類は流用のみ):
    - _load_npz / _board_from_grid / _delta_to_ojama_standard
      (scripts/label_exchange_outcome.py)
    - _delta_score_at_fire / _nearest_board (scripts/proto_net_threat.py)
    - _restrict_to_time_window / _classify_opp_coverage / OppCoverageStatus
      (scripts/measure_exchange_dynamics.py)
    - estimate_expected_net_damage / estimate_available_hands
      (scripts/measure_exchange_effectiveness.py)

attacker_chain_count は CSV 列 approx_fire_chains をそのまま使う
(label_exchange_outcome.py が既に current_max_chain 近似で計算済みのため、
盤面から再計算しない)。

使い方:
    PYTHONPATH=. python -m scripts.augment_exchange_labels_with_sim \\
        --input-csv data/indicators_v2/exchange_labels_regen_step0_2026-08-01.csv \\
        --npz-dir data/indicators_v2/boards_lean_regen_2026-07-31 \\
        --output data/indicators_v2/exchange_labels_regen_step0_aug_2026-08-01.csv

重い指標 (expected_fire_power の K=3,4 モンテカルロ) を1行ごとに呼ぶため、
イベント数が多い CSV では長時間かかる (行数に比例)。10分おき進捗ログ
(feedback_progress_notify_10min 準拠) を出す。

## 並列化 (--workers、2026-08-02 追加)
23,716件で423分かかり律速だったため、video_id単位で並列化した (feedback_
parallelize_evals_by_default)。既定 --workers=1 は旧実装と完全に同じコード
パス (_run_sequential) を通るため挙動不変。--workers>1 では動画をイベント
件数降順に貪欲(LPT)割当し、multiprocessing.Pool で並列処理する。

expected_fire_power の K=3,4 モンテカルロは盤面内容から決定論的に導出した
シードを使う (乱数不使用の厳密解ではなくMCだが、同一盤面には常に同一結果、
壁時計やプロセスIDに依存しない)。そのため --workers=1 と --workers>1 の
出力は理論上ビット一致する (行の計算順序・プロセスに依存する状態を持たない)。
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.board import Board  # noqa: E402
from src.indicators_v2 import expected_fire_power  # noqa: E402
from scripts.label_exchange_outcome import (  # noqa: E402
    NpzRecord,
    _board_from_grid,
    _delta_to_ojama_standard,
    _load_npz,
)
from scripts.proto_net_threat import _delta_score_at_fire, _nearest_board  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    OppCoverageStatus,
    _classify_opp_coverage,
    _restrict_to_time_window,
)
from scripts.measure_exchange_effectiveness import (  # noqa: E402
    estimate_available_hands,
    estimate_expected_net_damage,
)

# ============================
# 定数定義
# ============================
DEFAULT_INPUT_CSV = PROJ_ROOT / "data" / "indicators_v2" / "exchange_labels_regen_step0_2026-08-01.csv"
DEFAULT_NPZ_DIR = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_regen_2026-07-31"
DEFAULT_OUTPUT_PATH = PROJ_ROOT / "data" / "indicators_v2" / "exchange_labels_regen_step0_aug_2026-08-01.csv"

# 進捗ログ間隔 (feedback_progress_notify_10min 準拠)
PROGRESS_LOG_EVERY_SEC: float = 600.0

# sim_* 列が計算不能 (delta_score 欠損等) だった場合の埋め値
SIM_NAN: float = float("nan")

# ============================
# 並列化 (--workers、2026-08-02 追加)
# ============================
# 既定=1 (逐次処理、旧挙動と完全一致)。行単位の計算は独立だが npz キャッシュ
# (_VideoCache) を worker ごとに開き直すコストがあるため video_id 単位で分割する。
DEFAULT_WORKERS: int = 1

# worker内スレッド数制限 (feedback_subprocess_env_propagation: 環境変数は
# {**os.environ, ...} でマージし他の変数を消さない。project_collect_indicators_v2_perf
# の「cv2.setNumThreads(1)×N並列」と同じ思想=1プロセス1コアでBLAS等の
# オーバーサブスクリプションを防ぐ)。
WORKER_THREAD_ENV_VARS: tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
WORKER_THREAD_LIMIT: str = "1"

# 進捗をカウンタに反映するバッチサイズ (行ごとにロックを取ると重いためまとめる)
PROGRESS_REPORT_BATCH: int = 20


class _VideoCache:
    """1動画分の npz 由来データをまとめて保持する (side別 NpzRecord + Board化済み配列)。"""

    def __init__(self, records: list[NpzRecord]) -> None:
        self.by_side: dict[str, NpzRecord] = {r.side: r for r in records}
        # 側ごとの (t_sec, Board) 全フレームリスト (nearest match 用、1動画1回だけ構築)。
        self._boards_cache: dict[str, list[tuple[float, Board]]] = {}

    def boards_for_side(self, side: str) -> "list[tuple[float, Board]] | None":
        """side 側の全フレーム (t_sec, Board) リストを返す (遅延構築・キャッシュ)。"""
        if side not in self.by_side:
            return None
        if side not in self._boards_cache:
            rec = self.by_side[side]
            self._boards_cache[side] = [
                (float(t), _board_from_grid(g)) for t, g in zip(rec.t_sec, rec.grids)
            ]
        return self._boards_cache[side]


def _opp_side(fire_side: str) -> str:
    """発火側から見た相手側の文字列を返す。"""
    return "2P" if fire_side == "1P" else "1P"


# CSV の video_id 列は npz 内蔵フィールドそのまま ("video_c10" 等、
# scripts/label_exchange_outcome.py 経由で NpzRecord.video_id を格納した値)
# だが、npz ファイル名自体は接頭辞なし ("c10.npz") のため変換が必要
# (実データで確認済み: 2026-08-01)。
_VIDEO_ID_NPZ_PREFIX: str = "video_"


def _video_id_to_npz_stem(video_id: str) -> str:
    """CSV の video_id ("video_c10") を npz ファイル名の stem ("c10") に変換する。"""
    if video_id.startswith(_VIDEO_ID_NPZ_PREFIX):
        return video_id[len(_VIDEO_ID_NPZ_PREFIX):]
    return video_id


def _load_video_cache(video_id: str, npz_dir: Path) -> "_VideoCache | None":
    """1動画分の npz を読み込み _VideoCache を構築する (無ければ None + 警告ログ)。

    逐次実行 (--workers=1) と並列実行 (--workers>1) の両方から呼ばれる共通処理
    (元は main() 内に直書きされていたロジックをそのまま抽出、挙動は不変)。
    """
    npz_stem = _video_id_to_npz_stem(str(video_id))
    npz_path = npz_dir / f"{npz_stem}.npz"
    if not npz_path.exists():
        print(f"[WARN] npz が見つかりません: {npz_path} (video_id={video_id} 全行NaN)",
              file=sys.stderr)
        return None
    return _VideoCache(_load_npz(npz_path))


def _compute_sim_columns_for_row(
    row: "pd.Series",
    cache: _VideoCache,
    mode: str,
) -> "tuple[float, float, float]":
    """1発火イベント行分の (sim_k_hands, sim_expected_counter_ojama, sim_damage_score) を計算する。

    delta_score が取得できない (発火が npz 先頭フレームで直前有効スコアが
    存在しない等の境界ケース) 場合は 3値とも NaN を返す
    (誤って 0 と混同させないため)。
    """
    fire_side: str = row["fire_side"]
    opp_side = _opp_side(fire_side)
    t_fire = float(row["t_sec"])
    game_idx = int(row["game_idx"])
    approx_chains = float(row["approx_fire_chains"])

    fire_full = cache.by_side.get(fire_side)
    opp_full = cache.by_side.get(opp_side)
    if fire_full is None or opp_full is None:
        return SIM_NAN, SIM_NAN, SIM_NAN

    game_mask = fire_full.game_idx == game_idx
    if not game_mask.any():
        return SIM_NAN, SIM_NAN, SIM_NAN
    fire_t_sec = fire_full.t_sec[game_mask]
    fire_score = fire_full.score[game_mask]
    fire_grids = fire_full.grids[game_mask]

    fire_idx = int(np.argmin(np.abs(fire_t_sec - t_fire)))
    delta_score = _delta_score_at_fire(fire_score, fire_idx)
    if delta_score is None:
        return SIM_NAN, SIM_NAN, SIM_NAN
    attacker_ojama_sent = float(_delta_to_ojama_standard(delta_score))
    attacker_board_after_fire = _board_from_grid(fire_grids[fire_idx])

    # 相手の被覆状態 (連鎖中=応手不能 かどうか) を判定。
    # attacker 自身のこのゲームの実時刻範囲で opp_full (全ゲーム分) を絞り込む
    # (game_idx でなく実時刻で絞る、measure_exchange_dynamics.py と同じ設計)。
    own_game_start_t = float(fire_t_sec.min())
    own_game_end_t = float(fire_t_sec.max())
    opp_window = _restrict_to_time_window(opp_full, own_game_start_t, own_game_end_t)
    coverage_status = _classify_opp_coverage(t_fire, own_game_end_t, opp_window)

    # 相手盤面は (label_exchange_outcome.py と同じ) 時刻最近傍で復元する。
    opp_boards = cache.boards_for_side(opp_side)
    if opp_boards is None or len(opp_boards) == 0:
        return SIM_NAN, SIM_NAN, SIM_NAN
    opp_t_arr = np.array([t for t, _b in opp_boards])
    opp_board = _nearest_board(opp_t_arr, opp_boards, t_fire)

    k_hands = estimate_available_hands(approx_chains)
    damage_score = estimate_expected_net_damage(
        attacker_ojama_sent=attacker_ojama_sent,
        opp_board=opp_board,
        opp_coverage_status=coverage_status,
        attacker_chain_count=approx_chains,
        attacker_board_after_fire=attacker_board_after_fire,
        elapsed_sec=0.0,
        mode=mode,
    )
    # sim_expected_counter_ojama は estimate_expected_net_damage 内部の中間値
    # (attacker_ojama_sent - net_expected の逆算、二重計算を避けるため
    # OPP_CHAINING 分岐だけここでも明示的に再現する。関数内ロジックと同じ
    # 条件式を保つこと)。
    if coverage_status == OppCoverageStatus.OPP_CHAINING:
        expected_counter_ojama = 0.0
    else:
        result = expected_fire_power(opp_board, k_levels=(k_hands,), elapsed_sec=0.0)
        expected_counter_ojama = float(result.values[k_hands].raw)

    return float(k_hands), expected_counter_ojama, damage_score


# ============================
# 並列化: video_id単位の貪欲バランス割当 (--workers>1)
# ============================

def balance_videos_by_event_count(
    video_counts: "dict[str, int]", n_workers: int,
) -> list[list[str]]:
    """動画をイベント件数降順に貪欲(LPT: Longest Processing Time)割当する。

    件数が最も多い動画から順に、その時点で合計件数が最小の worker に割り当てる
    ことで worker 間の負荷 (行数) をバランスさせる。n_workers 個の video_id
    リストを返す (動画本数が worker 数より少なければ末尾の一部は空リスト)。
    """
    videos_desc = sorted(video_counts.items(), key=lambda kv: kv[1], reverse=True)
    assigned: list[list[str]] = [[] for _ in range(n_workers)]
    loads = [0] * n_workers
    for video_id, count in videos_desc:
        target = loads.index(min(loads))
        assigned[target].append(video_id)
        loads[target] += count
    return assigned


def _apply_worker_thread_limits() -> None:
    """worker内のBLAS等のスレッド数を1に制限する (1プロセス1コアでオーバー

    サブスクリプションを防ぐ、project_collect_indicators_v2_perf と同じ思想)。
    feedback_subprocess_env_propagation 準拠: 既存の環境変数を消さず
    {**os.environ, ...} でマージする。
    """
    os.environ.update({**os.environ, **{k: WORKER_THREAD_LIMIT for k in WORKER_THREAD_ENV_VARS}})


# Pool worker 内でのみ設定される進捗共有カウンタ (initializer 経由でグローバル登録)。
_worker_progress_counter: "multiprocessing.sharedctypes.Synchronized | None" = None


def _init_worker(progress_counter: "multiprocessing.sharedctypes.Synchronized") -> None:
    """Pool worker 初期化 (スレッド数制限 + 進捗共有カウンタの登録)。"""
    global _worker_progress_counter
    _worker_progress_counter = progress_counter
    _apply_worker_thread_limits()


def _process_video_chunk(
    task: "tuple[list[tuple[str, pd.DataFrame]], Path, str]",
) -> list[tuple[int, float, float, float]]:
    """1 worker が担当する動画群の sim_* 列をまとめて計算する。

    npz は担当動画ごとに1回だけ読み込む (_VideoCache 構築コストを分割数だけ
    払わないため)。進捗はグローバル共有カウンタに PROGRESS_REPORT_BATCH 件
    単位でまとめて反映する (行ごとのロック取得コストを避ける)。
    """
    video_subdfs, npz_dir, mode = task
    results: list[tuple[int, float, float, float]] = []
    pending = 0
    for video_id, sub_df in video_subdfs:
        cache = _load_video_cache(video_id, npz_dir)
        for idx, row in sub_df.iterrows():
            if cache is not None:
                k_hands, exp_counter, damage = _compute_sim_columns_for_row(row, cache, mode)
            else:
                k_hands, exp_counter, damage = SIM_NAN, SIM_NAN, SIM_NAN
            results.append((idx, k_hands, exp_counter, damage))
            pending += 1
            if pending >= PROGRESS_REPORT_BATCH and _worker_progress_counter is not None:
                with _worker_progress_counter.get_lock():
                    _worker_progress_counter.value += pending
                pending = 0
    if pending and _worker_progress_counter is not None:
        with _worker_progress_counter.get_lock():
            _worker_progress_counter.value += pending
    return results


def _build_parallel_tasks(
    df: pd.DataFrame, npz_dir: Path, mode: str, n_workers: int,
) -> "tuple[list[tuple], list[int]]":
    """video_id単位で貪欲バランス割当し、workerタスクのリストと件数内訳を作る。"""
    video_groups: dict[str, pd.DataFrame] = dict(iter(df.groupby("video_id", sort=False)))
    video_counts = {vid: len(g) for vid, g in video_groups.items()}
    assignments = balance_videos_by_event_count(video_counts, n_workers)
    tasks = [
        ([(vid, video_groups[vid]) for vid in vids], npz_dir, mode)
        for vids in assignments if vids
    ]
    task_counts = [sum(video_counts[vid] for vid in vids) for vids in assignments if vids]
    return tasks, task_counts


def _poll_and_log_progress(async_result, progress_counter, n_total: int) -> None:
    """完了待ちの間、完了イベント数を約10分おきにログする (feedback_progress_notify_10min)。"""
    start_t = time.monotonic()
    last_log_t = start_t
    while not async_result.ready():
        async_result.wait(timeout=10.0)
        now = time.monotonic()
        if now - last_log_t >= PROGRESS_LOG_EVERY_SEC:
            n_done = progress_counter.value
            elapsed_min = (now - start_t) / 60.0
            rate = n_done / max(1e-6, now - start_t)
            remain_sec = (n_total - n_done) / max(1e-6, rate)
            print(f"[PROGRESS] {n_done}/{n_total} 行完了 "
                  f"(経過{elapsed_min:.1f}分、残り約{remain_sec / 60.0:.1f}分)")
            last_log_t = now


def _merge_chunk_results(
    results_per_task: "list[list[tuple[int, float, float, float]]]", n_total: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """各workerの結果 [(idx,k_hands,exp_counter,damage),...] を元の行順の配列にマージする。"""
    sim_k_hands = np.full(n_total, SIM_NAN, dtype=np.float64)
    sim_expected_counter_ojama = np.full(n_total, SIM_NAN, dtype=np.float64)
    sim_damage_score = np.full(n_total, SIM_NAN, dtype=np.float64)
    for chunk in results_per_task:
        for idx, k_hands, exp_counter, damage in chunk:
            sim_k_hands[idx] = k_hands
            sim_expected_counter_ojama[idx] = exp_counter
            sim_damage_score[idx] = damage
    return sim_k_hands, sim_expected_counter_ojama, sim_damage_score


def _run_sequential(
    df: pd.DataFrame, npz_dir: Path, mode: str,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """--workers=1 (既定) の逐次実行。旧実装と完全に同じ処理内容 (挙動不変)。"""
    sim_k_hands = np.full(len(df), SIM_NAN, dtype=np.float64)
    sim_expected_counter_ojama = np.full(len(df), SIM_NAN, dtype=np.float64)
    sim_damage_score = np.full(len(df), SIM_NAN, dtype=np.float64)

    video_cache: "dict[str, _VideoCache | None]" = {}
    start_t = time.monotonic()
    last_log_t = start_t
    n_done = 0
    n_total = len(df)

    for video_id, group in df.groupby("video_id", sort=False):
        if video_id not in video_cache:
            video_cache[video_id] = _load_video_cache(video_id, npz_dir)
        cache = video_cache[video_id]

        for idx, row in group.iterrows():
            if cache is not None:
                k_hands, exp_counter, damage = _compute_sim_columns_for_row(row, cache, mode)
            else:
                k_hands, exp_counter, damage = SIM_NAN, SIM_NAN, SIM_NAN
            sim_k_hands[idx] = k_hands
            sim_expected_counter_ojama[idx] = exp_counter
            sim_damage_score[idx] = damage
            n_done += 1

            now = time.monotonic()
            if now - last_log_t >= PROGRESS_LOG_EVERY_SEC:
                elapsed_min = (now - start_t) / 60.0
                rate = n_done / max(1e-6, now - start_t)
                remain_sec = (n_total - n_done) / max(1e-6, rate)
                print(f"[PROGRESS] {n_done}/{n_total} 行完了 "
                      f"(経過{elapsed_min:.1f}分、残り約{remain_sec / 60.0:.1f}分)")
                last_log_t = now

    return sim_k_hands, sim_expected_counter_ojama, sim_damage_score


def _run_parallel(
    df: pd.DataFrame, npz_dir: Path, mode: str, n_workers: int,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """--workers>1 時の並列実行本体 (video_id単位の貪欲バランス割当 + 進捗集計)。"""
    tasks, task_counts = _build_parallel_tasks(df, npz_dir, mode, n_workers)
    print(f"[INFO] workers={n_workers}  動画{df['video_id'].nunique()}本 -> "
          f"{len(tasks)}タスクに割当 (各タスク件数: {task_counts})")
    progress_counter = multiprocessing.Value("L", 0)
    with multiprocessing.Pool(
        processes=len(tasks), initializer=_init_worker, initargs=(progress_counter,),
    ) as pool:
        async_result = pool.map_async(_process_video_chunk, tasks)
        _poll_and_log_progress(async_result, progress_counter, len(df))
        results_per_task = async_result.get()
    return _merge_chunk_results(results_per_task, len(df))


def _parse_args() -> "argparse.Namespace":
    """CLI 引数をパースする。"""
    import argparse
    parser = argparse.ArgumentParser(description="exchange_labels CSV に修正シミュ特徴量(Step5)を付与する")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV,
                         help=f"入力 exchange_labels CSV (既定: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR,
                         help=f"入力 npz ディレクトリ (既定: {DEFAULT_NPZ_DIR})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH,
                         help=f"出力 CSV パス (既定: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--mode", type=str, default="precise", choices=["precise", "fast"],
                         help="estimate_expected_net_damage に渡す mode (既定: precise)")
    parser.add_argument("--limit", type=int, default=None,
                         help="先頭 N 行だけ処理する (動作確認・タイミング測定用、既定は全件)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help=f"並列worker数 (既定={DEFAULT_WORKERS}=逐次実行、旧挙動と完全一致。"
                              "2以上でvideo_id単位の並列処理)")
    return parser.parse_args()


def main() -> None:
    """メイン処理。"""
    warnings.filterwarnings("ignore")
    args = _parse_args()
    input_csv: Path = args.input_csv
    npz_dir: Path = args.npz_dir
    output_path: Path = args.output

    if not input_csv.exists():
        print(f"[ERROR] 入力CSVが見つかりません: {input_csv}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(input_csv)
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"[INFO] 入力 {len(df)} 行 ({input_csv})")

    start_t = time.monotonic()
    if args.workers <= 1:
        sim_k_hands, sim_expected_counter_ojama, sim_damage_score = _run_sequential(
            df, npz_dir, args.mode,
        )
    else:
        sim_k_hands, sim_expected_counter_ojama, sim_damage_score = _run_parallel(
            df, npz_dir, args.mode, args.workers,
        )

    df["sim_k_hands"] = sim_k_hands
    df["sim_expected_counter_ojama"] = sim_expected_counter_ojama
    df["sim_damage_score"] = sim_damage_score

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    total_min = (time.monotonic() - start_t) / 60.0
    print(f"[DONE] {len(df)} 行を {output_path} に保存しました (所要{total_min:.1f}分)")
    valid = df["sim_damage_score"].notna()
    print(f"  sim_damage_score 有効 (非NaN): {valid.sum()}/{len(df)} "
          f"({valid.mean() * 100:.1f}%)")
    if valid.any():
        print(f"  sim_k_hands 分布: {df.loc[valid, 'sim_k_hands'].value_counts().to_dict()}")
        print(f"  sim_expected_counter_ojama mean={df.loc[valid, 'sim_expected_counter_ojama'].mean():.2f}")
        print(f"  sim_damage_score mean={df.loc[valid, 'sim_damage_score'].mean():.3f}")


if __name__ == "__main__":
    main()
