"""連鎖イベント統計収集スクリプト。

各動画から連鎖イベントを抽出し、
  - 連鎖数
  - 連鎖所要時間 (秒) ← STABLE→非STABLE→STABLE の実占有時間
  - 生成スコア (点) および お邪魔換算
を CSV に記録する。
別途 --aggregate モードで全 CSV を結合し曲線 A/B を算出・プロット。

【連鎖時間の計測方式 (v2)】
ChainEvent.end_sec は「消去検出フレーム(=発火翌フレーム)」なので
trigger_sec との差はほぼ 1 フレーム(≈0.033s)になる(旧バグ)。
正しい実占有時間は:
  duration = (次にSTABLEへ復帰した時刻) - (発火時刻 = STABLE→非STABLEへ遷移した時刻)
を使う。collect_one 内の _ChainTimer クラスで side ごとに追跡する。

使い方 (1 本処理):
    python -m scripts.collect_chain_stats \\
        --video data/frames/video_29.mp4 \\
        --video-id video_29 \\
        --start-sec 140 --end-sec 700

使い方 (集計モード):
    python -m scripts.collect_chain_stats --aggregate
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

# プロジェクトルートを import path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_state_machine import BoardState, NON_STABLE_STATES  # noqa: E402
from src.chain_detector import ChainEvent  # noqa: E402
from src.chain import ChainSimulator  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline, SideResult  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402

# ============================
# 定数
# ============================

# 出力ディレクトリ (プロジェクトルートからの相対)
OUTPUT_DIR: Path = Path("data/indicators_v2/chain_stats")

# 認識解像度
TARGET_W: int = 1920
TARGET_H: int = 1080
DEFAULT_FPS: float = 30.0

# 有効イベントの最小連鎖数フィルタ:
# chain_count >= 2 OR gen_score > 0 が必要。
# 単発消しかつスコアゼロ = 検出ゆらぎのノイズとして除外する。
MIN_CHAIN_COUNT_STRICT: int = 2   # これ以上なら gen_score 問わず採用
MIN_GEN_SCORE_FOR_SINGLE: int = 1  # 1 連鎖でも score > 0 なら採用

# 連鎖所要時間の上限ガード (秒):
# CHAIN_MAX_HOLD_SEC=5.0s (recognition_pipeline) + マージン。
# 10s を超える連鎖は実ゲームでは存在しない。
MAX_DURATION_SEC: float = 10.0

# 発火タイムアウト: NON-STABLE のままこの秒数を超えたら記録を破棄する。
# pipeline の CHAIN_MAX_HOLD_SEC=5.0 に余裕を持たせた値。
FIRE_TIMEOUT_SEC: float = 8.0

# スコア → お邪魔換算レート (公式: 70 点 = 1 個)
SCORE_TO_OJAMA_RATE: int = 70

# CSV 列定義 (順序固定)
# gen_score: ChainEvent.total_score (点)
# gen_ojama: round(gen_score / SCORE_TO_OJAMA_RATE) (個)
CSV_COLUMNS: tuple[str, ...] = (
    "video_id", "side", "chain_count", "duration_sec",
    "gen_score", "gen_ojama", "t_start",
)


# ============================
# 連鎖時間追跡: _ChainTimer
# ============================

@dataclass
class _PendingFire:
    """発火中の連鎖イベント (STABLE 復帰待ち) を保持する。"""

    event: ChainEvent        # ChainEvent (chain_count / total_score 参照用)
    fire_sec: float          # STABLE → 非STABLE に遷移した時刻 (= 実発火開始)


@dataclass
class _ChainTimer:
    """side ごとに STABLE 遷移を追跡し、実占有 duration を計算する。

    設計:
      - STABLE → 非STABLE 遷移フレームで chain_event があれば _PendingFire に記録。
      - 非STABLE → STABLE 遷移フレームで pending を確定 → duration = 復帰秒 - 発火秒。
      - タイムアウト(FIRE_TIMEOUT_SEC)超過 or MENU 復帰の場合は破棄。
    """

    side: str
    _prev_state: BoardState = field(default=BoardState.MENU)
    _pending: Optional["_PendingFire"] = field(default=None)

    def feed(
        self,
        current_state: BoardState,
        t_sec: float,
        chain_event: Optional[ChainEvent],
    ) -> Optional[dict]:
        """現フレームの state / time / chain_event を受け取り、確定レコードを返す。

        Returns:
            確定した場合: {"fire_sec", "stable_return_sec", "duration_sec", "event"}
            未確定の場合: None
        """
        result: Optional[dict] = None
        prev = self._prev_state

        # STABLE → 非STABLE 遷移 = 発火開始
        if (
            prev == BoardState.STABLE
            and current_state in NON_STABLE_STATES
            and chain_event is not None
        ):
            # 前フレームの STABLE 時刻は持っていないため t_sec を発火開始とみなす。
            # (遷移フレーム自体が最初の非STABLE フレームなので誤差は ≤1 フレーム)
            self._pending = _PendingFire(event=chain_event, fire_sec=t_sec)

        # 非STABLE → STABLE 遷移 = 連鎖終了・STABLE 復帰
        elif (
            prev in NON_STABLE_STATES
            and current_state == BoardState.STABLE
            and self._pending is not None
        ):
            duration = t_sec - self._pending.fire_sec
            result = {
                "fire_sec": self._pending.fire_sec,
                "stable_return_sec": t_sec,
                "duration_sec": duration,
                "event": self._pending.event,
            }
            self._pending = None

        # タイムアウト判定: 非STABLE が続きすぎたら破棄
        elif (
            current_state in NON_STABLE_STATES
            and self._pending is not None
            and (t_sec - self._pending.fire_sec) > FIRE_TIMEOUT_SEC
        ):
            print(
                f"[SKIP/{self.side}] t={self._pending.fire_sec:.1f}s 発火タイムアウト "
                f"({FIRE_TIMEOUT_SEC}s超) → 破棄",
                file=sys.stderr,
            )
            self._pending = None

        # MENU 復帰 = 試合終了 → 保留中を破棄
        elif current_state == BoardState.MENU and self._pending is not None:
            self._pending = None

        self._prev_state = current_state
        return result


# ============================
# 連鎖イベント処理
# ============================

def _chain_count_from_event(event: ChainEvent) -> int:
    """ChainEvent から連鎖数を返す。

    chain_count が 0 のイベントはシミュレーション失敗を示すため 0 を返す。
    """
    return int(event.chain_count)


def _gen_score_from_event(event: ChainEvent) -> int:
    """ChainEvent から生成スコア (点) を返す。

    total_score = 全消しボーナス持越し込みの実効スコア。
    """
    return int(event.total_score)


def _gen_ojama_from_score(gen_score: int) -> int:
    """スコア (点) をお邪魔個数に換算する。

    公式: 70 点 = 1 個 (端数切り捨て)。
    """
    return gen_score // SCORE_TO_OJAMA_RATE


def _is_valid_event(
    chain_count: int, duration_sec: float, gen_score: int,
) -> tuple[bool, str]:
    """連鎖イベントが有効かどうかを検証する。

    有効条件:
      - chain_count >= MIN_CHAIN_COUNT_STRICT (2 連鎖以上)
        OR gen_score > 0 (1 連鎖でも火力あり)
      - duration_sec > 0 かつ <= MAX_DURATION_SEC
      - gen_score >= 0 (負は異常)

    Returns:
        (is_valid, reason) — 無効な場合は reason に除外理由を記載。
    """
    # ノイズフィルタ: 単発消し (chain=1) かつ gen_score=0 は除外
    if chain_count < MIN_CHAIN_COUNT_STRICT and gen_score < MIN_GEN_SCORE_FOR_SINGLE:
        return False, (
            f"chain_count={chain_count} < {MIN_CHAIN_COUNT_STRICT} "
            f"かつ gen_score={gen_score}=0 (単発消し/ノイズ)"
        )
    if duration_sec <= 0.0:
        return False, f"duration_sec={duration_sec:.3f} <= 0 (負・ゼロ)"
    if duration_sec > MAX_DURATION_SEC:
        return False, f"duration_sec={duration_sec:.1f} > {MAX_DURATION_SEC} (異常値)"
    if gen_score < 0:
        return False, f"gen_score={gen_score} < 0 (異常値)"
    return True, ""


# ============================
# メイン処理: 1 動画
# ============================

def _make_pipeline(video_id: str) -> "RecognitionPipeline":
    """連鎖統計用 RecognitionPipeline を生成する。"""
    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3,
        load_score_ocr=True,
        enable_chain_tracker=True,
        temporal_smoothing=1,
        load_next_detector=True,
        force_in_match=True,
    )
    if hasattr(pipeline, "set_video_id"):
        pipeline.set_video_id(video_id)
    return pipeline


def _process_side_timer(
    video_id: str,
    side_label: str,
    side_res: "SideResult",
    timer: _ChainTimer,
    t_sec: float,
    records: list[dict[str, object]],
    noise_count_ref: list[int],
) -> None:
    """1 side 分のタイマー更新と確定レコード追記を行う。

    noise_count_ref は [count] の単要素リストで、呼び出し元と共有する。
    """
    confirmed = timer.feed(
        current_state=side_res.state,
        t_sec=t_sec,
        chain_event=side_res.chain_event,
    )
    if confirmed is None:
        return

    ev: ChainEvent = confirmed["event"]
    duration_sec: float = confirmed["duration_sec"]
    chain_count = _chain_count_from_event(ev)
    gen_score = _gen_score_from_event(ev)
    gen_ojama = _gen_ojama_from_score(gen_score)

    valid, reason = _is_valid_event(chain_count, duration_sec, gen_score)
    if not valid:
        noise_count_ref[0] += 1
        print(
            f"[SKIP] {video_id}/{side_label} t={confirmed['fire_sec']:.1f}s "
            f"chain={chain_count} score={gen_score} dur={duration_sec:.3f}s → {reason}",
            file=sys.stderr,
        )
        return

    records.append({
        "video_id": video_id,
        "side": side_label,
        "chain_count": chain_count,
        "duration_sec": round(duration_sec, 3),
        "gen_score": gen_score,
        "gen_ojama": gen_ojama,
        "t_start": round(confirmed["fire_sec"], 3),
    })


def collect_one(
    video_path: Path,
    video_id: str,
    start_sec: float = 0.0,
    end_sec: float = 0.0,
) -> list[dict[str, object]]:
    """1 動画を処理して連鎖イベントレコードのリストを返す。

    連鎖時間の計測方式 (v2):
      旧: ChainEvent.end_sec - trigger_sec (≈1フレーム = バグ)
      新: _ChainTimer で STABLE→非STABLE→STABLE 遷移を追跡し実占有時間を計測。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けない: {video_path}", file=sys.stderr)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps) if start_sec > 0.0 else 0
    end_frame = min(total_frames, int(end_sec * fps)) if end_sec > 0.0 else total_frames
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    n_frames = max(0, end_frame - start_frame)
    print(
        f"[INFO] {video_id}: frames {start_frame}–{end_frame} "
        f"({n_frames / fps:.0f}s), fps={fps:.1f}",
        file=sys.stderr,
    )

    pipeline = _make_pipeline(video_id)
    ojama_tracker = OjamaAccountingTracker()
    ojama_tracker.reset()
    prev_state_p1 = BoardState.MENU
    prev_state_p2 = BoardState.MENU
    timer_1p = _ChainTimer(side="1P")
    timer_2p = _ChainTimer(side="2P")
    records: list[dict[str, object]] = []
    noise_ref = [0]

    for local_i in range(n_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        fi = start_frame + local_i
        t_sec = fi / fps
        result = pipeline.update(fi, t_sec, frame)

        ojama_tracker.on_state_transition(
            "p1", prev_state_p1, result.p1.state, result.p1.score, t_sec,
        )
        ojama_tracker.on_state_transition(
            "p2", prev_state_p2, result.p2.state, result.p2.score, t_sec,
        )
        prev_state_p1 = result.p1.state
        prev_state_p2 = result.p2.state

        _process_side_timer(video_id, "1P", result.p1, timer_1p, t_sec, records, noise_ref)
        _process_side_timer(video_id, "2P", result.p2, timer_2p, t_sec, records, noise_ref)

    cap.release()
    print(
        f"[INFO] {video_id}: 有効={len(records)}件 除外(ノイズ)={noise_ref[0]}件",
        file=sys.stderr,
    )
    return records


def save_csv(records: list[dict[str, object]], out_path: Path) -> None:
    """連鎖イベントレコードを CSV に書き出す。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    print(f"[INFO] CSV 保存: {out_path} ({len(records)} 行)", file=sys.stderr)


# ============================
# 集計モード
# ============================

def aggregate() -> None:
    """OUTPUT_DIR 内の全 CSV を結合し曲線 A/B を算出してプロットする。"""
    import numpy as np
    from scipy.optimize import curve_fit  # type: ignore[import]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Meiryo フォント (WSL パス)
    meiryo_path = "/mnt/c/Windows/Fonts/meiryo.ttc"
    if Path(meiryo_path).exists():
        font_manager.fontManager.addfont(meiryo_path)
        plt.rcParams["font.family"] = "Meiryo"
    else:
        print(f"[WARN] Meiryo not found at {meiryo_path}, フォールバック使用", file=sys.stderr)

    # 全 CSV 結合
    all_records: list[dict[str, str]] = []
    csv_files = sorted(OUTPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] CSV が見つからない: {OUTPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    for csv_path in csv_files:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_records.extend(reader)

    print(f"[INFO] 合計 {len(all_records)} 行 ({len(csv_files)} ファイル)", file=sys.stderr)

    chain_counts = np.array([int(r["chain_count"]) for r in all_records])
    durations = np.array([float(r["duration_sec"]) for r in all_records])
    # gen_ojama 列は v2 CSV では round(gen_score / 70)。旧 CSV も同列名なので互換。
    gen_ojamaS = np.array([int(r["gen_ojama"]) for r in all_records])
    # gen_score 列は v2 CSV のみ存在。旧 CSV では gen_ojama * 70 で近似。
    gen_scores = np.array([
        int(r["gen_score"]) if "gen_score" in r else int(r["gen_ojama"]) * SCORE_TO_OJAMA_RATE
        for r in all_records
    ])

    # ============================
    # 曲線 A: 連鎖数 → 時間
    # ============================
    _plot_curve_a(chain_counts, durations, OUTPUT_DIR / "curveA_time.png")

    # ============================
    # 曲線 B: 連鎖数 → 生成お邪魔 + 指数フィット (お邪魔単位)
    # ============================
    _plot_curve_b(chain_counts, gen_ojamaS, OUTPUT_DIR / "curveB_power.png")


def _plot_curve_a(
    chain_counts: "np.ndarray",
    durations: "np.ndarray",
    out_path: Path,
) -> None:
    """曲線 A: 連鎖数ごとの median/mean/n を集計してプロット。"""
    import numpy as np
    import matplotlib.pyplot as plt

    unique_chains = sorted(set(chain_counts.tolist()))
    stats: list[dict[str, float]] = []
    # 頑健化: 1フレーム潰れ(≈0.033s)の spurious 重複行を除外してから集計。
    # _ChainTimer が STABLE 復帰を取りこぼした行が median を 0.03s に張り付かせるため。
    spurious = 0.05
    print("\n=== 曲線 A: 連鎖数 → 時間 (spurious<=%.2fs除外) ===" % spurious, file=sys.stderr)
    print(f"{'chain':>6} {'n':>5} {'median':>8} {'mean':>8}", file=sys.stderr)
    for c in unique_chains:
        mask = chain_counts == c
        vals = durations[mask]
        vals = vals[vals > spurious]
        n = int(vals.size)
        if n == 0:
            continue
        med = float(np.median(vals))
        mean = float(np.mean(vals))
        stats.append({"chain": float(c), "n": float(n), "median": med, "mean": mean})
        print(f"{c:>6} {n:>5} {med:>8.2f}s {mean:>8.2f}s", file=sys.stderr)

    xs = [s["chain"] for s in stats]
    medians = [s["median"] for s in stats]
    means = [s["mean"] for s in stats]
    ns = [s["n"] for s in stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, medians, "o-", label="median", linewidth=2)
    ax.plot(xs, means, "s--", label="mean", linewidth=1.5, alpha=0.7)
    for x, n in zip(xs, ns):
        ax.annotate(f"n={n}", (x, 0.2), ha="center", fontsize=7, color="gray")
    ax.set_xlabel("連鎖数")
    ax.set_ylabel("所要時間 (秒)")
    ax.set_title("曲線 A: 連鎖数 → 実発火時間")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[INFO] 曲線 A 保存: {out_path}", file=sys.stderr)


def _exp_func(x: "np.ndarray", a: float, b: float) -> "np.ndarray":
    """指数フィット関数: a * exp(b * x)。"""
    import numpy as np
    return a * np.exp(b * x)


def _plot_curve_b(
    chain_counts: "np.ndarray",
    gen_ojamaS: "np.ndarray",
    out_path: Path,
) -> None:
    """曲線 B: 連鎖数 → 生成お邪魔個数 (= score/70) + 指数フィット。"""
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    unique_chains = sorted(set(chain_counts.tolist()))
    xs_data: list[float] = []
    ys_data: list[float] = []
    print("\n=== 曲線 B: 連鎖数 → 生成お邪魔個数 (score/70) ===", file=sys.stderr)
    print(f"{'chain':>6} {'n':>5} {'mean(個)':>10} {'median(個)':>11}", file=sys.stderr)
    for c in unique_chains:
        mask = chain_counts == c
        vals = gen_ojamaS[mask]
        n = int(mask.sum())
        mean = float(np.mean(vals))
        med = float(np.median(vals))
        xs_data.append(float(c))
        ys_data.append(mean)
        print(f"{c:>6} {n:>5} {mean:>10.1f} {med:>11.1f}", file=sys.stderr)

    xs_arr = np.array(xs_data)
    ys_arr = np.array(ys_data)

    # 指数フィット (お邪魔個数単位)
    fit_params: tuple[float, float] | None = None
    try:
        # gen_ojama=0 の点はフィット除外 (0 はスコアゼロ = ノイズ)
        mask_nonzero = ys_arr > 0
        if mask_nonzero.sum() >= 3:
            popt, _ = curve_fit(
                _exp_func, xs_arr[mask_nonzero], ys_arr[mask_nonzero],
                p0=[1.0, 0.5], maxfev=5000,
            )
            fit_params = (float(popt[0]), float(popt[1]))
            print(
                f"\n指数フィット(お邪魔個数): ojama ≈ {fit_params[0]:.4f}"
                f" * exp({fit_params[1]:.4f} * chain)",
                file=sys.stderr,
            )
        else:
            print("[WARN] フィット可能点が不足 (非ゼロ < 3)", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] 指数フィット失敗: {e}", file=sys.stderr)

    # プロット
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(xs_arr, ys_arr, alpha=0.6, label="mean gen_ojama (個=score/70)")
    if fit_params is not None:
        xs_fit = np.linspace(float(xs_arr.min()), float(xs_arr.max()), 100)
        ys_fit = _exp_func(xs_fit, *fit_params)
        ax.plot(
            xs_fit, ys_fit, "r-",
            label=f"exp fit: {fit_params[0]:.3f}*exp({fit_params[1]:.3f}*x)",
            linewidth=2,
        )
    ax.set_xlabel("連鎖数")
    ax.set_ylabel("生成お邪魔個数 (平均, score/70)")
    ax.set_title("曲線 B: 連鎖数 → 生成お邪魔個数 (指数フィット)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[INFO] 曲線 B 保存: {out_path}", file=sys.stderr)


# ============================
# CLI エントリポイント
# ============================

def _build_parser() -> argparse.ArgumentParser:
    """引数パーサを構築する。"""
    p = argparse.ArgumentParser(description="連鎖イベント統計収集スクリプト")
    p.add_argument("--video", type=Path, help="入力動画パス")
    p.add_argument("--video-id", type=str, help="動画 ID (CSV の video_id 列)")
    p.add_argument("--start-sec", type=float, default=0.0, help="処理開始秒")
    p.add_argument("--end-sec", type=float, default=0.0, help="処理終了秒 (0=全長)")
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="集計モード: OUTPUT_DIR の全 CSV を結合して曲線をプロット",
    )
    return p


def main() -> None:
    """メインエントリポイント。"""
    parser = _build_parser()
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
        return

    if args.video is None:
        parser.error("--aggregate を指定しない場合は --video が必要です")
    if args.video_id is None:
        # video_id 省略時はファイルステムから自動決定
        args.video_id = args.video.stem

    records = collect_one(
        video_path=args.video,
        video_id=args.video_id,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )

    out_path = OUTPUT_DIR / f"{args.video_id}.csv"
    save_csv(records, out_path)


if __name__ == "__main__":
    main()
