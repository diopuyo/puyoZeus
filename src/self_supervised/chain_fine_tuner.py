"""ChainFineTuner: VideoChainTracker の閾値を擬似ラベルで最適化する.

実装方針 (gradient-free grid search):
    chain_validator.py が emit する擬似ラベルは
        input_data["before_board_grid"]  (CHAIN 開始直前の盤面)
        input_data["score_jump"]         (CHAIN 期間中の score 増分)
        input_data["duration_frames"]    (CHAIN state 持続フレーム数)
        label["chain_count"]             (ChainSimulator から導いた真の連鎖数)
        label["total_erased"]            (真の消去ぷよ数)
        metadata["chain_count_match"]    (検出側の chain_count と一致するか)
        metadata["score_match"] / ["duration_match"]

    VideoChainTracker は本質的に
        ・直前 stable 盤面と現在の puyo 数の差 (= drop)
        ・drop >= erasure_min_drop なら検出開始
        ・snapshot_lookback フレーム前の盤面を発火盤面とみなす
    で動く。

    本 fine-tuner は擬似ラベルを再現用 fixture として grid search を行う:
        各 sample について label.total_erased >= 候補 erasure_min_drop なら
            tracker は「検出する」 (= 1)
        さらに simulate(before_board) の chain_count >= 1 が ground truth
        accuracy = 一致率 (TP+TN)/N

    duration_frames が高品質 (= chain_count*EXPECTED_FRAMES_PER_CHAIN 近傍) な
    sample に絞って評価する optional フィルタも持つ (`min_confidence`).

    snapshot_lookback はサンプルから直接最適化できないため、
    duration_frames の中央値ベースで heuristic に fix する補助評価のみ。

出力:
    data/verify/chain_tracker_calibration.json:
        {
            "erasure_min_drop": int,
            "snapshot_lookback": int,
            "n_samples": int,
            "accuracy_before": float,
            "accuracy_after": float,
            "best_params": {...},
        }

依存:
    - ChainSimulator (再シミュレーション)
    - PseudoLabelSample
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, Board, VALID_COLORS
from src.chain import ChainSimulator
from src.chain_detector import ERASURE_MIN_DROP, SNAPSHOT_LOOKBACK
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.self_supervised.online_fine_tuner import OnlineFineTuner
from src.self_supervised.pseudo_label import (
    COMPONENT_CHAIN,
    PseudoLabelSample,
)

# ============================
# 定数
# ============================

# 既定の calibration 出力先
DEFAULT_CALIBRATION_PATH: Path = Path("data/verify/chain_tracker_calibration.json")

# fine_tune に必要な最小サンプル数
MIN_SAMPLES_FOR_FIT: int = 16

# サンプルを高信頼に絞る最低 confidence
DEFAULT_MIN_CONFIDENCE: float = 0.0  # 既定では全サンプル使用 (validator 側で 0.7+)

# tunable パラメータ候補 (探索空間)
DEFAULT_ERASURE_GRID: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10)
DEFAULT_LOOKBACK_GRID: tuple[int, ...] = (1, 2, 3, 4)

# backup suffix
DEFAULT_BACKUP_SUFFIX: str = ".bak"


# ============================
# データ
# ============================


@dataclass(frozen=True)
class _ReplaySample:
    """grid search 用の前処理済 sample.

    Attributes:
        true_chain_count: ChainSimulator が出した chain_count (ground truth).
        true_total_erased: 同上.
        duration_frames: CHAIN state 持続フレーム数.
        chain_count_match: validator が detection と sim を比較した結果.
        confidence: 元 PseudoLabelSample の confidence.
    """

    true_chain_count: int
    true_total_erased: int
    duration_frames: int
    chain_count_match: bool
    confidence: float


@dataclass
class ChainFineTuneMetrics:
    """fine_tune の戻り値 (dict 化用)."""

    n_samples: int
    accuracy_before: float | None
    accuracy_after: float | None
    best_params: dict[str, Any] = field(default_factory=dict)
    grid_results: list[dict[str, Any]] = field(default_factory=list)
    skipped_reason: str | None = None


# ============================
# Fine Tuner 本体
# ============================


class ChainFineTuner(OnlineFineTuner):
    """VideoChainTracker の閾値 (erasure_min_drop / snapshot_lookback) を grid search で最適化."""

    def __init__(
        self,
        calibration_path: Path | str = DEFAULT_CALIBRATION_PATH,
        erasure_grid: Iterable[int] = DEFAULT_ERASURE_GRID,
        lookback_grid: Iterable[int] = DEFAULT_LOOKBACK_GRID,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        backup_suffix: str = DEFAULT_BACKUP_SUFFIX,
    ) -> None:
        self._path = Path(calibration_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._erasure_grid = tuple(int(x) for x in erasure_grid)
        self._lookback_grid = tuple(int(x) for x in lookback_grid)
        if not self._erasure_grid:
            raise ValueError("erasure_grid must be non-empty")
        if not self._lookback_grid:
            raise ValueError("lookback_grid must be non-empty")
        if min_confidence < 0.0 or min_confidence > 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self._min_confidence = float(min_confidence)
        self._backup_suffix = str(backup_suffix)
        self._backup_done: bool = False
        # ChainSimulator は再利用 (キャッシュ効果)
        # 幽霊連鎖ルール (2026-08-10 本番ON採用): production_config.py が単一情報源。
        self._simulator = ChainSimulator(
            exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED,
        )

    # ------------------------------------------------------------------
    # OnlineFineTuner API
    # ------------------------------------------------------------------

    def fine_tune(
        self, samples: list[PseudoLabelSample],
    ) -> dict[str, Any]:
        """擬似ラベル群から最適な (erasure_min_drop, snapshot_lookback) を探索する."""
        replay = self._prepare_samples(samples)
        if len(replay) < MIN_SAMPLES_FOR_FIT:
            return ChainFineTuneMetrics(
                n_samples=len(replay),
                accuracy_before=None,
                accuracy_after=None,
                skipped_reason="not_enough_samples",
            ).__dict__
        baseline = self._evaluate(replay, ERASURE_MIN_DROP, SNAPSHOT_LOOKBACK)
        best_acc, best_params, grid_results = self._grid_search(replay)
        # backup → 上書き保存
        if not self._backup_done and self._path.exists():
            self._make_backup()
        self._save_calibration(
            best_params, baseline, best_acc, len(replay),
        )
        metrics = ChainFineTuneMetrics(
            n_samples=len(replay),
            accuracy_before=float(baseline),
            accuracy_after=float(best_acc),
            best_params=best_params,
            grid_results=grid_results,
        )
        return metrics.__dict__

    def rollback(self) -> None:
        """backup から復元 (無ければ calibration を削除)."""
        backup_path = self._path.with_suffix(
            self._path.suffix + self._backup_suffix,
        )
        if backup_path.exists():
            shutil.copy2(backup_path, self._path)
            return
        if self._path.exists():
            try:
                self._path.unlink()
            except OSError:
                pass
        self._backup_done = False

    # ------------------------------------------------------------------
    # 内部 helper
    # ------------------------------------------------------------------

    def _prepare_samples(
        self, samples: list[PseudoLabelSample],
    ) -> list[_ReplaySample]:
        """擬似ラベルから replay 用の軽量タプル列を作る."""
        out: list[_ReplaySample] = []
        for s in samples:
            if s.component != COMPONENT_CHAIN:
                continue
            if s.confidence < self._min_confidence:
                continue
            inp = s.input_data if isinstance(s.input_data, dict) else None
            lab = s.label if isinstance(s.label, dict) else None
            meta = s.metadata if isinstance(s.metadata, dict) else {}
            if inp is None or lab is None:
                continue
            grid = inp.get("before_board_grid")
            if not isinstance(grid, list):
                continue
            board = _grid_to_board(grid)
            if board is None:
                continue
            try:
                sim = self._simulator.simulate(board)
            except Exception:
                continue
            duration = int(inp.get("duration_frames", 0))
            cnt_match = bool(meta.get("chain_count_match", False))
            out.append(_ReplaySample(
                true_chain_count=int(sim.chain_count),
                true_total_erased=int(sim.total_erased),
                duration_frames=duration,
                chain_count_match=cnt_match,
                confidence=float(s.confidence),
            ))
        return out

    def _grid_search(
        self, replay: list[_ReplaySample],
    ) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
        """erasure_grid × lookback_grid の格子を全探索する.

        Returns:
            (best_accuracy, best_params, all_grid_results)
        """
        results: list[dict[str, Any]] = []
        best_acc = -1.0
        best_params: dict[str, Any] = {}
        for emin in self._erasure_grid:
            for lookback in self._lookback_grid:
                acc = self._evaluate(replay, emin, lookback)
                results.append({
                    "erasure_min_drop": int(emin),
                    "snapshot_lookback": int(lookback),
                    "accuracy": float(acc),
                })
                if acc > best_acc:
                    best_acc = acc
                    best_params = {
                        "erasure_min_drop": int(emin),
                        "snapshot_lookback": int(lookback),
                    }
        return best_acc, best_params, results

    @staticmethod
    def _evaluate(
        replay: list[_ReplaySample],
        erasure_min_drop: int,
        snapshot_lookback: int,
    ) -> float:
        """ある (erasure_min_drop, lookback) における整合率を評価.

        ground truth: true_chain_count >= 1 (ChainSimulator 出力).
        prediction:   true_total_erased >= erasure_min_drop (drop で検出).
        snapshot_lookback は accuracy には直接影響しないが、
            duration_frames との整合に補助 score を加える:
            duration が短い (lookback*2 frame 以下) sample は不確定として
            重みを 0.5 に減らす.
        """
        if not replay:
            return 0.0
        weighted_correct = 0.0
        weighted_total = 0.0
        # lookback が大きいほど短い CHAIN 期間に弱くなるので減衰補正
        threshold_short = max(1, snapshot_lookback)
        for r in replay:
            gt = r.true_chain_count >= 1
            pred = r.true_total_erased >= erasure_min_drop
            w = 1.0 if r.duration_frames > threshold_short else 0.5
            weighted_total += w
            if gt == pred:
                weighted_correct += w
        if weighted_total <= 0.0:
            return 0.0
        return weighted_correct / weighted_total

    # ------------------------------------------------------------------
    # 保存系
    # ------------------------------------------------------------------

    def _save_calibration(
        self,
        best_params: dict[str, Any],
        accuracy_before: float,
        accuracy_after: float,
        n_samples: int,
    ) -> None:
        """calibration を atomic に書き出す."""
        data: dict[str, Any] = {
            "erasure_min_drop": int(best_params.get(
                "erasure_min_drop", ERASURE_MIN_DROP,
            )),
            "snapshot_lookback": int(best_params.get(
                "snapshot_lookback", SNAPSHOT_LOOKBACK,
            )),
            "n_samples": int(n_samples),
            "accuracy_before": float(accuracy_before),
            "accuracy_after": float(accuracy_after),
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._path)
        self._backup_done = True

    def _make_backup(self) -> None:
        backup_path = self._path.with_suffix(
            self._path.suffix + self._backup_suffix,
        )
        try:
            shutil.copy2(self._path, backup_path)
        except OSError:
            pass


# ============================
# helpers
# ============================


def _grid_to_board(grid: list[list[int]]) -> Board | None:
    """JSON-serialized grid を Board へ変換 (UNKNOWN セルは EMPTY 扱い).

    UNKNOWN (=10) は ChainSimulator が許容しないため、
    シミュレーション目的で空セルへ置換する.
    """
    if len(grid) != BOARD_ROWS:
        return None
    cleaned: list[list[int]] = []
    for row in grid:
        if len(row) != BOARD_COLS:
            return None
        cleaned_row: list[int] = []
        for v in row:
            try:
                vi = int(v)
            except (TypeError, ValueError):
                return None
            # UNKNOWN は EMPTY 化、その他無効値は EMPTY へ防御的に変換
            if vi == 10 or vi not in VALID_COLORS:
                cleaned_row.append(COLOR_EMPTY)
            else:
                cleaned_row.append(vi)
        cleaned.append(cleaned_row)
    try:
        return Board.from_list(cleaned)
    except ValueError:
        return None


__all__ = [
    "DEFAULT_BACKUP_SUFFIX",
    "DEFAULT_CALIBRATION_PATH",
    "DEFAULT_ERASURE_GRID",
    "DEFAULT_LOOKBACK_GRID",
    "DEFAULT_MIN_CONFIDENCE",
    "MIN_SAMPLES_FOR_FIT",
    "ChainFineTuneMetrics",
    "ChainFineTuner",
]
