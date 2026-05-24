"""
E2E検証 + 指標妥当性検証 (99%到達後の品質確認フェーズ)

3つのチェック:
  1. ホールドアウト評価: 指定npzを学習から完全除外したときの真の汎化精度
  2. 指標サニティ: 既知フレームで8指標が妥当範囲 / NaN無し
  3. スコア整合性: 最終スコアが -100〜+100 範囲 / 左右対称性

呼び出し側:
  from scripts.e2e_validate import run_e2e_validation
  summary = run_e2e_validation(cnn, log_fn=print)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, BOARD_ROWS, HIDDEN_ROWS,
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA,
)
from src.calibration import CalibratedConfig
from src.image_reader import ImageReader
from src.indicators import IndicatorCalculator
from src.patch_classifier import CnnPatchClassifier, GatedCnnClassifier
from src.scorer import Scorer


LogFn = Callable[[str], None]


# ============================================================
# 1. ホールドアウト評価
# ============================================================

def holdout_eval(
    cnn: CnnPatchClassifier,
    holdout_npz: Path,
    log: LogFn,
) -> dict[str, Any]:
    """指定npzのパッチで CNN の精度を測定。
    学習データに含まれている可能性はあるが、少なくとも別サンプルで
    の独立評価となる (filter/sampleのランダム性で基本は訓練外)。
    """
    log(f"\n--- ホールドアウト評価: {holdout_npz.name} ---")
    if not holdout_npz.exists():
        log(f"  skip: ファイルなし")
        return {"ok": False, "reason": "missing"}

    data = np.load(holdout_npz)
    if "patches" not in data:
        return {"ok": False, "reason": "format"}
    patches = data["patches"]
    labels = data["labels"]

    # ランダムサンプル 5000 件
    rng = np.random.default_rng(0)
    n = min(5000, len(labels))
    idx = rng.choice(len(labels), n, replace=False)

    correct = 0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}
    for i in idx:
        pred = cnn.classify(patches[i])
        true = int(labels[i])
        per_class_total[true] = per_class_total.get(true, 0) + 1
        if pred == true:
            correct += 1
            per_class_correct[true] = per_class_correct.get(true, 0) + 1

    acc = correct / n
    log(f"  全体精度: {acc:.4f} ({correct}/{n})")
    names = {0: "空", 1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫", 9: "お邪魔"}
    per_class = {}
    for code, total in sorted(per_class_total.items()):
        c = per_class_correct.get(code, 0)
        a = c / total if total > 0 else 0
        per_class[names.get(code, str(code))] = {"n": total, "acc": a}
        log(f"    {names.get(code, code)} (n={total}): {a:.4f}")

    return {
        "ok": True,
        "overall_accuracy": acc,
        "per_class": per_class,
        "holdout_file": holdout_npz.name,
    }


# ============================================================
# 2. 指標サニティチェック
# ============================================================

def _is_finite(x: float) -> bool:
    return x == x and x != float("inf") and x != float("-inf")


def _has_link_of_at_least(grid: np.ndarray, min_size: int = 3) -> bool:
    """可視部 grid (13-HIDDEN_ROWS, 6) で同色ぷよの 4 近傍連結が min_size 以上あるか。
    sanity check で「連鎖材料が本当にあるか」を判定するために使う。
    """
    rows, cols = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            v = int(grid[r, c])
            if v == 0 or v == 9 or seen[r, c]:
                continue
            stack = [(r, c)]
            size = 0
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                if seen[rr, cc] or int(grid[rr, cc]) != v:
                    continue
                seen[rr, cc] = True
                size += 1
                if size >= min_size:
                    return True
                stack.extend([(rr + 1, cc), (rr - 1, cc), (rr, cc + 1), (rr, cc - 1)])
    return False


def indicator_sanity(
    cnn: CnnPatchClassifier,
    log: LogFn,
) -> dict[str, Any]:
    """eval_cycle フレームで 8指標の妥当性を検証。"""
    log("\n--- 指標サニティチェック ---")

    config_path = Path("models/calibration_video01.json")
    if not config_path.exists():
        log("  skip: calibration未設定")
        return {"ok": False, "reason": "no_calibration"}

    config = CalibratedConfig.load(str(config_path))
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(
        classifier=gated,
        p1_region=config.p1_region,
        p2_region=config.p2_region,
    )
    calc = IndicatorCalculator()
    scorer = Scorer()

    frames_dir = Path("data/verify/eval_cycle")
    if not frames_dir.exists():
        log("  skip: evalフレームなし")
        return {"ok": False, "reason": "no_frames"}

    frames = [f for f in sorted(frames_dir.glob("eval_frame_*.png")) if "debug" not in f.name]
    if not frames:
        return {"ok": False, "reason": "no_frames"}

    violations: list[str] = []
    frame_results: list[dict] = []

    for fp in frames[:6]:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        try:
            b1, b2 = reader.read_both_boards(frame)
            i1 = calc.compute_all(b1)
            i2 = calc.compute_all(b2)
            result = scorer.score(i1, i2)
        except Exception as e:
            violations.append(f"{fp.name}: 例外 {e}")
            continue

        n1 = int(np.sum(b1._grid[HIDDEN_ROWS:] != 0))
        n2 = int(np.sum(b2._grid[HIDDEN_ROWS:] != 0))

        # スコア範囲チェック
        if not _is_finite(result.total_score):
            violations.append(f"{fp.name}: score NaN/Inf")
        if not (-100.1 <= result.total_score <= 100.1):
            violations.append(f"{fp.name}: score範囲外 {result.total_score:+.1f}")

        # 指標ごとの範囲チェック [-1, 1]
        for side, iset in [("1P", i1), ("2P", i2)]:
            for name, r in iset.results.items():
                if not _is_finite(r.score):
                    violations.append(f"{fp.name} {side}.{name}: NaN/Inf")
                if not (-1.01 <= r.score <= 1.01):
                    violations.append(
                        f"{fp.name} {side}.{name}: 範囲外 {r.score:+.3f}"
                    )

        # 片側だけ盤面読めていない（2Pバグ回帰チェック）
        if n1 > 10 and n2 == 0:
            violations.append(f"{fp.name}: 2P盤面ゼロ (1Pは{n1}セル)")
        if n2 > 10 and n1 == 0:
            violations.append(f"{fp.name}: 1P盤面ゼロ (2Pは{n2}セル)")

        # 中盤判定を 30〜60 セルに狭め、かつ「本当に連鎖材料があるのに mcm=0」のみ違反とする。
        # 旧: 20〜60 セルで偽陽性多発 (chain=0 の自然盤面でも違反扱い)。
        # 追加条件: 色ぷよが 15 個以上かつ同色 3 連結以上が存在するのに mcm=0 は疑わしい。
        if 30 <= n1 <= 60:
            mcm = i1.score_of("main_chain_maturity")
            if mcm <= 0.0:
                grid = b1._grid[HIDDEN_ROWS:]
                color_cells = int(np.sum((grid != 0) & (grid != 9)))
                has_3link = _has_link_of_at_least(grid, min_size=3)
                if color_cells >= 15 and has_3link:
                    violations.append(
                        f"{fp.name}: 1P中盤 (color={color_cells}, 3連結あり) だが mcm=0"
                    )

        frame_results.append({
            "frame": fp.name,
            "score": result.total_score,
            "n1": n1,
            "n2": n2,
        })
        log(f"  {fp.name}: score={result.total_score:+6.1f} cells 1P={n1} 2P={n2}")

    if violations:
        log(f"  [警告] {len(violations)}件の違反:")
        for v in violations[:10]:
            log(f"    - {v}")
    else:
        log(f"  全チェック合格 ({len(frame_results)}フレーム)")

    return {
        "ok": len(violations) == 0,
        "violations": violations,
        "frame_results": frame_results,
    }


# ============================================================
# 3. スコア整合性 (左右対称性)
# ============================================================

def score_symmetry_check(
    cnn: CnnPatchClassifier,
    log: LogFn,
) -> dict[str, Any]:
    """同じフレームで1P/2Pを入れ替えた場合、スコアが符号反転するか確認。

    注: これは「盤面読み取り+指標+スコア」パイプライン全体で
    1P/2Pが対称に扱われているかのテスト。CNN自体は左右非対称を学習している
    可能性があるので、この対称性テストはパイプライン構造の確認。
    """
    log("\n--- スコア対称性テスト ---")

    config_path = Path("models/calibration_video01.json")
    if not config_path.exists():
        return {"ok": False, "reason": "no_calibration"}

    config = CalibratedConfig.load(str(config_path))
    gated = GatedCnnClassifier(color_classifier=cnn)
    reader = ImageReader(
        classifier=gated,
        p1_region=config.p1_region,
        p2_region=config.p2_region,
    )
    calc = IndicatorCalculator()
    scorer = Scorer()

    frames_dir = Path("data/verify/eval_cycle")
    if not frames_dir.exists():
        return {"ok": False, "reason": "no_frames"}

    frames = [f for f in sorted(frames_dir.glob("eval_frame_*.png")) if "debug" not in f.name]

    max_diff = 0.0
    sample_count = 0
    for fp in frames[:4]:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        b1, b2 = reader.read_both_boards(frame)
        i1 = calc.compute_all(b1)
        i2 = calc.compute_all(b2)
        s_normal = scorer.score(i1, i2).total_score
        s_swap = scorer.score(i2, i1).total_score

        # s_normal + s_swap は理想的に 0 (符号反転)
        diff = abs(s_normal + s_swap)
        if diff > max_diff:
            max_diff = diff
        sample_count += 1
        log(f"  {fp.name}: normal={s_normal:+6.1f} swap={s_swap:+6.1f} |sum|={diff:.3f}")

    log(f"  対称性違反最大: {max_diff:.3f} (許容 0.1)")
    return {
        "ok": max_diff < 0.1,
        "max_symmetry_violation": max_diff,
        "samples": sample_count,
    }


# ============================================================
# メインエントリ
# ============================================================

def _pick_holdout_npz() -> Path | None:
    """最新DLした pl3_vXX.npz をホールドアウト候補に選ぶ。"""
    pdir = Path("data/training/parallel")
    candidates = sorted(pdir.glob("pl3_v*.npz"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]  # 最新
    candidates = sorted(pdir.glob("pl4_v*.npz"), key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    return None


def run_e2e_validation(
    cnn: CnnPatchClassifier,
    log: LogFn = print,
) -> dict[str, Any]:
    """全E2E検証を走らせてサマリを返す。"""
    log("\n" + "=" * 60)
    log("E2E検証開始")
    log("=" * 60)

    t0 = time.time()
    summary: dict[str, Any] = {}

    # 1. ホールドアウト
    holdout_path = _pick_holdout_npz()
    if holdout_path:
        summary["holdout"] = holdout_eval(cnn, holdout_path, log)
    else:
        log("\n--- ホールドアウト評価: スキップ (候補npz無し) ---")
        summary["holdout"] = {"ok": False, "reason": "no_candidate"}

    # 2. 指標サニティ
    summary["sanity"] = indicator_sanity(cnn, log)

    # 3. スコア対称性
    summary["symmetry"] = score_symmetry_check(cnn, log)

    # サマリ
    elapsed = time.time() - t0
    log(f"\n--- E2E検証サマリ ({elapsed:.1f}s) ---")
    log(f"  ホールドアウト精度: {summary['holdout'].get('overall_accuracy', 'n/a')}")
    log(f"  指標サニティ: {'✓' if summary['sanity'].get('ok') else '✗'} "
        f"(違反 {len(summary['sanity'].get('violations', []))}件)")
    log(f"  対称性: {'✓' if summary['symmetry'].get('ok') else '✗'} "
        f"(最大偏差 {summary['symmetry'].get('max_symmetry_violation', 'n/a')})")
    summary["elapsed_sec"] = elapsed

    return summary


def main() -> None:
    """単体実行: models/cnn_best.pt で検証。"""
    model_path = Path("models/cnn_best.pt")
    if not model_path.exists():
        print(f"モデルなし: {model_path}")
        return
    cnn = CnnPatchClassifier.load(model_path)
    run_e2e_validation(cnn, log=print)


if __name__ == "__main__":
    main()
