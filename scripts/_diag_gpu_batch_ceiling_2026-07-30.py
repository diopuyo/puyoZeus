"""
GPU バッチ分類の上限速度を実測する診断スクリプト (2026-07-30)。

目的: 「×印判定+HSV統計をCNNの1パスに統合しGPUへ寄せる」計画の前提
(= GPUバッチ推論が1ms以下で収まるか)を、本体改修前に実測で確認する。

**src/ は一切変更しない**。既存 CnnPatchClassifierLarge をそのままロードし、
以下 4 点を測定する:
    1. 156セル (1P+2P × 78セル) を 1 バッチで forward した場合の実時間
       (転送込み / forward単体)。
    2. 現状相当の 10 チャンク分割呼び出しとの比較。
    3. 盤面ROI一括転送 + GPU側 unfold/reshape によるパッチ切り出し方式。
    4. VRAM 使用量、および出力クラス数を増やした場合の速度影響。

推測でなく実測値のみを報告する (= 本スクリプトの標準出力がそのまま報告根拠)。
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.patch_classifier import (
    CLASS_INDEX_TO_COLOR,
    NUM_CLASSES,
    PATCH_RESIZE_H,
    PATCH_RESIZE_W,
    CnnPatchClassifierLarge,
)

# ============================
# 定数定義 (マジックナンバー排除)
# ============================

# 本番既定モデル (recognition_pipeline.py:1924 DEFAULT_CNN_MODEL_PATH)
DEFAULT_MODEL_PATH: Path = Path("models/cnn_phase_b_large_v2.pt")

# 1P + 2P それぞれ 13行×6列 = 78 セル、 合計 156 (課題指定値)
CELLS_PER_BOARD: int = 78
NUM_BOARDS: int = 2
TOTAL_CELLS: int = CELLS_PER_BOARD * NUM_BOARDS  # = 156

# セル切り出し前の想定ネイティブ画素サイズ (1920x1080 盤面領域の実測相当値)
NATIVE_CELL_PX: int = 64

# 盤面 ROI 一括転送方式: 13行×6列 セルをネイティブ画素で並べた領域
ROI_HEIGHT_PX: int = NATIVE_CELL_PX * 13   # 832
ROI_WIDTH_PX: int = NATIVE_CELL_PX * 6     # 384

# ベンチ設定
WARMUP_ITERS: int = 30
BENCH_ITERS: int = 200          # 「100回以上」要件を満たす
CHUNK_COUNT_CURRENT: int = 10   # プロファイルで観測された現状の分割数

# 判定閾値 (課題指定)
JUDGE_OK_MS: float = 1.0
JUDGE_NG_MS: float = 10.0

RNG_SEED: int = 42


@dataclass
class TimingResult:
    """複数回計測の分布要約。"""

    median_ms: float
    p90_ms: float
    mean_ms: float

    @classmethod
    def from_samples(cls, samples_sec: list[float]) -> "TimingResult":
        ms = sorted(s * 1000.0 for s in samples_sec)
        n = len(ms)
        p90_idx = min(n - 1, int(round(0.9 * (n - 1))))
        return cls(
            median_ms=statistics.median(ms),
            p90_ms=ms[p90_idx],
            mean_ms=statistics.fmean(ms),
        )

    def __str__(self) -> str:
        return (
            f"median={self.median_ms:.3f}ms p90={self.p90_ms:.3f}ms "
            f"mean={self.mean_ms:.3f}ms"
        )


def _make_synthetic_patches(n: int, seed: int) -> list[np.ndarray]:
    """乱数 BGR パッチ (ネイティブ画素サイズ) を n 枚生成する。

    実画像でなくとも forward 速度の実測には影響しない (形状のみが速度を
    決める)。 CPU 前処理コスト (cv2.resize/cvtColor) は既存プロファイルで
    別途計測済みのため、本スクリプトでは GPU 側 (転送+forward) の切り分けに
    専念する。
    """
    rng = np.random.default_rng(seed)
    return [
        rng.integers(0, 256, size=(NATIVE_CELL_PX, NATIVE_CELL_PX, 3), dtype=np.uint8)
        for _ in range(n)
    ]


def _build_cpu_batch_tensor(
    classifier: CnnPatchClassifierLarge, patches: list[np.ndarray],
) -> torch.Tensor:
    """既存 _patch_to_tensor を使い CPU 上で (N, 6, H, W) バッチテンソルを組む。

    このタイミングは計測対象外 (CPU 前処理コストは別枠) にするため、
    呼び出し側で warmup/計測ループの外に置くこと。
    """
    tensors = [classifier._patch_to_tensor(p)[0] for p in patches]
    return torch.stack(tensors)


def bench_single_batch(
    classifier: CnnPatchClassifierLarge,
    cpu_batch: torch.Tensor,
    device: str,
    iters: int,
) -> tuple[TimingResult, TimingResult, TimingResult, TimingResult]:
    """156セル 1バッチ forward: 転送込み全体 / forward単体 / 転送単体 / 取出単体。"""
    model = classifier._model
    full_samples: list[float] = []
    forward_samples: list[float] = []
    transfer_samples: list[float] = []
    retrieve_samples: list[float] = []

    for i in range(WARMUP_ITERS + iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_batch = cpu_batch.to(device, non_blocking=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            logits = model(gpu_batch)
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        t3 = time.perf_counter()
        if i < WARMUP_ITERS:
            continue
        transfer_samples.append(t1 - t0)
        forward_samples.append(t2 - t1)
        retrieve_samples.append(t3 - t2)
        full_samples.append(t3 - t0)
    del probs
    return (
        TimingResult.from_samples(full_samples),
        TimingResult.from_samples(forward_samples),
        TimingResult.from_samples(transfer_samples),
        TimingResult.from_samples(retrieve_samples),
    )


def bench_chunked(
    classifier: CnnPatchClassifierLarge,
    cpu_batch: torch.Tensor,
    device: str,
    n_chunks: int,
    iters: int,
) -> TimingResult:
    """現状相当: 156セルを n_chunks 分割し、都度 .to()+forward+.cpu() する。

    プロファイルで観測された「10バッチ/フレーム」呼び出し実態を模擬する。
    """
    model = classifier._model
    chunks = list(torch.chunk(cpu_batch, n_chunks, dim=0))
    full_samples: list[float] = []

    for i in range(WARMUP_ITERS + iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for chunk in chunks:
            gpu_chunk = chunk.to(device, non_blocking=False)
            with torch.no_grad():
                logits = model(gpu_chunk)
            _ = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i < WARMUP_ITERS:
            continue
        full_samples.append(t1 - t0)
    return TimingResult.from_samples(full_samples)


def bench_roi_unfold(
    classifier: CnnPatchClassifierLarge,
    device: str,
    iters: int,
) -> TimingResult:
    """盤面ROI一括転送 + GPU側 unfold/reshape によるパッチ切り出し方式。

    転送は 1P/2P 分をまとめて 1 回 (batch 次元 2) にし、GPU 上で
    unfold によりセル単位 (NATIVE_CELL_PX四方) に分割、adaptive_avg_pool2d
    で既存モデル入力サイズ (PATCH_RESIZE_H x PATCH_RESIZE_W) にダウンサンプル
    してから forward する。CPU 側の cv2.resize/cvtColor 切り出しが丸ごと不要
    になる方式の実測値。

    注意: HSV チャンネルは本来 CPU cvtColor で作るが、本ベンチは速度測定が
    目的のため BGR を複製した dummy 6ch (既存 INPUT_CHANNELS=6 と同形状) を
    使う。色相計算コストを除いた「転送+切り出し+forward」の下限を測る。
    """
    model = classifier._model
    rng = np.random.default_rng(RNG_SEED)
    # (NUM_BOARDS, 6ch, ROI_HEIGHT_PX, ROI_WIDTH_PX) の CPU テンソルを 1 回だけ生成
    roi_np = rng.integers(
        0, 256,
        size=(NUM_BOARDS, CnnPatchClassifierLarge.INPUT_CHANNELS, ROI_HEIGHT_PX, ROI_WIDTH_PX),
        dtype=np.uint8,
    ).astype(np.float32) / 255.0
    cpu_roi = torch.from_numpy(roi_np)

    full_samples: list[float] = []
    for i in range(WARMUP_ITERS + iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_roi = cpu_roi.to(device, non_blocking=False)
        # unfold で (NUM_BOARDS, 6, 13, 6, cell, cell) 相当をパッチ化
        patches = gpu_roi.unfold(2, NATIVE_CELL_PX, NATIVE_CELL_PX)
        patches = patches.unfold(3, NATIVE_CELL_PX, NATIVE_CELL_PX)
        # (NUM_BOARDS * 13 * 6, 6, cell, cell) に reshape
        n_rows, n_cols = patches.shape[2], patches.shape[3]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            NUM_BOARDS * n_rows * n_cols,
            CnnPatchClassifierLarge.INPUT_CHANNELS,
            NATIVE_CELL_PX, NATIVE_CELL_PX,
        )
        resized = torch.nn.functional.adaptive_avg_pool2d(
            patches, (PATCH_RESIZE_H, PATCH_RESIZE_W),
        )
        with torch.no_grad():
            logits = model(resized)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        t2 = time.perf_counter()
        if i < WARMUP_ITERS:
            continue
        full_samples.append(t2 - t0)
    return TimingResult.from_samples(full_samples)


def bench_extra_output_classes(
    base_classifier: CnnPatchClassifierLarge,
    cpu_batch: torch.Tensor,
    device: str,
    extra_classes: int,
    iters: int,
) -> TimingResult:
    """出力クラス数を増やした場合 (X印クラス追加相当) の forward 単体速度。

    最終 Linear 層のみ out_features を NUM_CLASSES+extra_classes に差し替えた
    複製モデルを作り、同一バッチで forward 単体時間を比較する。
    """
    import copy
    modified = copy.deepcopy(base_classifier._model)
    last_linear = modified[-1]
    assert isinstance(last_linear, torch.nn.Linear)
    new_linear = torch.nn.Linear(
        last_linear.in_features, NUM_CLASSES + extra_classes,
    )
    modified[-1] = new_linear
    modified = modified.to(device)
    modified.eval()

    gpu_batch = cpu_batch.to(device)
    forward_samples: list[float] = []
    for i in range(WARMUP_ITERS + iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = modified(gpu_batch)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i < WARMUP_ITERS:
            continue
        forward_samples.append(t1 - t0)
    return TimingResult.from_samples(forward_samples)


def measure_vram_mib(device: str) -> float:
    """直近の torch 操作による GPU メモリ最大割当量 (MiB) を返す。"""
    if device != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def _load_classifier(model_path: Path, device: str) -> CnnPatchClassifierLarge:
    """既存モデルファイルをロードし device に配置する (src/ 未変更でロード)。"""
    classifier = CnnPatchClassifierLarge.load(model_path)
    classifier.to_device(device)
    return classifier


def _print_gpu_context_note() -> None:
    """GPU/CPU 競合状況の注記 (nvidia-smi は別途手動確認前提、ここでは torch 視点のみ)。"""
    print("[注記] 本測定は他プロセスとのCPU/GPU競合下で実行 (330ジョブ収集14並列 実行中の可能性)。")
    print("       絶対値は競合の影響を受け得るが、相対比較 (単一バッチ vs 分割 vs ROI一括) は有効。")


def main() -> None:
    """全ベンチを実行し結果を標準出力に報告する。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}")
    print(f"model={DEFAULT_MODEL_PATH}  patch_input=({PATCH_RESIZE_H}x{PATCH_RESIZE_W}x6ch)  "
          f"num_classes={NUM_CLASSES}  classes={CLASS_INDEX_TO_COLOR}")
    _print_gpu_context_note()

    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {DEFAULT_MODEL_PATH}")

    classifier = _load_classifier(DEFAULT_MODEL_PATH, device)

    patches = _make_synthetic_patches(TOTAL_CELLS, RNG_SEED)
    cpu_batch = _build_cpu_batch_tensor(classifier, patches)
    print(f"\nbatch shape = {tuple(cpu_batch.shape)}  (TOTAL_CELLS={TOTAL_CELLS})")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("\n=== 1. 単一バッチ156 forward (転送込み/単体) ===")
    full, forward_only, transfer_only, retrieve_only = bench_single_batch(
        classifier, cpu_batch, device, BENCH_ITERS,
    )
    print(f"  全体(転送+forward+取出): {full}")
    print(f"  forward単体            : {forward_only}")
    print(f"  転送単体 (.to)         : {transfer_only}")
    print(f"  取出単体 (.cpu+numpy)  : {retrieve_only}")
    vram_single = measure_vram_mib(device)

    print(f"\n=== 2. 現状相当 {CHUNK_COUNT_CURRENT} チャンク分割 ===")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    chunked = bench_chunked(classifier, cpu_batch, device, CHUNK_COUNT_CURRENT, BENCH_ITERS)
    print(f"  10チャンク合計: {chunked}")
    print(f"  1バッチとの差分 (median): {chunked.median_ms - full.median_ms:+.3f}ms")

    print("\n=== 3. 盤面ROI一括転送 + GPU側unfold方式 ===")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    roi = bench_roi_unfold(classifier, device, BENCH_ITERS)
    print(f"  ROI={NUM_BOARDS}x6ch x {ROI_HEIGHT_PX}x{ROI_WIDTH_PX}px")
    print(f"  全体(転送+unfold+forward+取出): {roi}")
    vram_roi = measure_vram_mib(device)

    print("\n=== 4. 出力クラス数増加の影響 (X印クラス追加相当) ===")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    extra2 = bench_extra_output_classes(classifier, cpu_batch, device, 2, BENCH_ITERS)
    print(f"  +2クラス forward単体: {extra2}")
    print(f"  現行forward単体との差分 (median): {extra2.median_ms - forward_only.median_ms:+.4f}ms")

    print("\n=== VRAM使用量 ===")
    print(f"  単一バッチ156 (peak allocated): {vram_single:.2f} MiB / 8188 MiB")
    print(f"  ROI一括方式 (peak allocated)  : {vram_roi:.2f} MiB / 8188 MiB")

    print("\n=== 判定 ===")
    print(f"  閾値: OK<= {JUDGE_OK_MS}ms, NG>= {JUDGE_NG_MS}ms")
    verdict = (
        "計画成立 (OK)" if full.median_ms <= JUDGE_OK_MS
        else "前提崩壊 (NG)" if full.median_ms >= JUDGE_NG_MS
        else "境界域 (要再検討)"
    )
    print(f"  単一バッチ156 全体 median={full.median_ms:.3f}ms → {verdict}")


if __name__ == "__main__":
    main()
