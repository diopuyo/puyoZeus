"""Phase Z-2: cell 単位の検出漏れ・色誤認を補正する Refiner。

Phase Z-1 (半自動 GT ツール) の suspicious 解析で判明した CNN の弱点
を補正する。3 戦略を統合:

1. **EmRecovery**: recognized=EM/?? かつ HSV 中央が高彩度 (S≥60 V≥70)
   なら、HSV 主要色を採用 (検出漏れ救済)
2. **OjmRecovery**: recognized=EM/?? かつ低彩度+中域 V (S<55, 110≤V≤220)
   なら OJM 採用
3. **HsvVote**: recognized も HSV も puyo 色だが異なる場合、HSV の彩度が
   高ければ HSV 採用 (色 swap 救済)

設計上の注意:
    - HSV classifier への依存は HybridClassifier 経由 (内部の `_hsv` 属性)
    - 連鎖中フレーム (is_chain=True) ではこの Refiner はスキップ推奨
      (ChainSimulator 予測を尊重するため)
    - 統合先は StatePipeline.extract、CNN 分類後・他補正レイヤー前
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.board import (
    BOARD_COLS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN, HIDDEN_ROWS, Board,
)


# Z-1.7 で実証された閾値 (em_but_saturated)
EM_RECOVERY_S_MIN: int = 60
EM_RECOVERY_V_MIN: int = 70

# Z-1.8 で実証された閾値 (em_but_grayish → OJM)
# Z-3F: S=37-40 の OJM 漏れ救済のため OJM_S_MAX を 50 に緩和
OJM_RECOVERY_S_MAX: int = 50
OJM_RECOVERY_V_MIN: int = 145
OJM_RECOVERY_V_MAX: int = 210

# HsvVote: HSV を信頼するための最低彩度 (色 swap で HSV 側が確信を持つ場合のみ)
# Z-3K: 100→80 に緩和。env 修正後の真 sweep で +0.099pt (改善 9 動画 / 悪化 0) を
# 確認済の最初の改善 variant。HsvVote 発動範囲が広がり puyo の色 swap 補正が増える。
HSV_VOTE_S_MIN: int = 80


@dataclass(frozen=True)
class RefineStats:
    """1 board あたりの補正適用件数。デバッグ用。"""
    em_recovered: int = 0
    ojm_recovered: int = 0
    hsv_voted: int = 0


class CellRecoveryRefiner:
    """cell 単位の検出漏れ・色誤認を補正。

    Usage:
        refiner = CellRecoveryRefiner(hsv_classifier)
        new_board, stats = refiner.refine(frame, region, board)
    """

    def __init__(
        self,
        hsv_classifier,
        em_s_min: int = EM_RECOVERY_S_MIN,
        em_v_min: int = EM_RECOVERY_V_MIN,
        ojm_s_max: int = OJM_RECOVERY_S_MAX,
        ojm_v_min: int = OJM_RECOVERY_V_MIN,
        ojm_v_max: int = OJM_RECOVERY_V_MAX,
        hsv_vote_s_min: int = HSV_VOTE_S_MIN,
        enable_ojm_recovery: bool = False,
    ) -> None:
        self._hsv = hsv_classifier
        # default 閾値 (v18_m03 で 99.923% を達成した値)
        self._em_s_min = int(em_s_min)
        self._em_v_min = int(em_v_min)
        self._ojm_s_max = int(ojm_s_max)
        self._ojm_v_min = int(ojm_v_min)
        self._ojm_v_max = int(ojm_v_max)
        self._hsv_vote_s_min = int(hsv_vote_s_min)
        self._enable_ojm_recovery = bool(enable_ojm_recovery)
        # Z-3H: 動画別 BG calibrate されているか
        self._calibrated: bool = False

    def refine(
        self,
        frame: np.ndarray,
        region,
        board: Board,
        is_chain: bool = False,
        hsv_full: np.ndarray | None = None,
    ) -> tuple[Board, RefineStats]:
        """frame と region から cell ごとに HSV 分析し、3 戦略で補正。

        Args:
            is_chain: 連鎖中・相殺エフェクト中なら True。物理違反強制補正
                (airborne EM / empty_in_stack 補完) をスキップする。
                連鎖アニメ中の puyo 表示乱れで真の puyo を消失させないため。
            hsv_full: Z-3C 高速化。事前計算した frame 全体 HSV (cv2.cvtColor)。
                None なら refine 内で計算。複数 region (P1/P2) で再利用する場合
                は呼び出し側で 1 度だけ計算して渡す。
        """
        out = board.copy()
        em_count = 0
        ojm_count = 0
        vote_count = 0
        # Z-3C: HSV 全体変換は 1 度だけ (P1/P2 で共有可能)
        if hsv_full is None:
            hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 上から順に補正 (airborne チェックは下端から行うため、補正 → 検証)
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                rec_color = int(out.get(row, col))
                patch = self._extract_inner(frame, region, row, col)
                if patch is None or patch.size == 0:
                    continue
                # Z-3C: hsv_full からスライス (cvtColor 不要)
                s_mean, v_mean = self._hsv_mean_from_full(
                    hsv_full, region, row, col,
                )
                # 戦略 1 + 2: recognized=EM/?? を救う
                if rec_color in (COLOR_EMPTY, COLOR_UNKNOWN):
                    new_color = self._recover_em(patch, s_mean, v_mean)
                    if new_color != rec_color:
                        out.set(row, col, new_color)
                        if new_color == COLOR_OJAMA:
                            ojm_count += 1
                        else:
                            em_count += 1
                    continue
                # 戦略 3: HsvVote — recognized も HSV も puyo 色で異なる場合
                # Z-3F: OJM 判定の cell でも、HSV が高彩度 puyo 色を返せば補正
                # (OJM↔PUR 混同 5 件のような CNN 固定誤認に対応)
                if s_mean >= self._hsv_vote_s_min:
                    hsv_color = self._classify_hsv(patch)
                    if (hsv_color not in (
                            COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA,
                    ) and hsv_color != rec_color):
                        out.set(row, col, hsv_color)
                        vote_count += 1
                # Z-3F: PUR→EM 過剰補正対策
                # CNN が puyo と判定したが、低彩度 + 中域 V (= 真の EM 背景特徴) の
                # 場合は EM に補正。HSV ベースで CNN 誤検出を是正。
                # Z-3G: 連鎖中は puyo 保護優先で skip (相殺エフェクト対策)
                elif (s_mean < 50 and v_mean < 100
                        and rec_color != COLOR_OJAMA
                        and not is_chain):
                    out.set(row, col, COLOR_EMPTY)
                    vote_count += 1
        # airborne チェック (1): puyo cell の直下が EM なら物理的に不可能 → 取り消し
        # 隠し段 (HIDDEN_ROWS=1) は対象外、可視領域だが最下段以外で適用
        # 連鎖中でも、補正で puyo 化した cell が浮遊なら戻す (over-correction 防止)
        reverted = self._revert_airborne(out, board)
        em_count -= reverted

        # Z-3G: 連鎖中・相殺エフェクト中は airborne 強制 EM / empty_in_stack 補完を skip
        # 連鎖アニメで puyo 表示が乱れる瞬間に真の puyo を消失させるバグ対策
        if is_chain:
            return out, RefineStats(
                em_recovered=em_count,
                ojm_recovered=ojm_count,
                hsv_voted=vote_count,
            )

        # airborne 強制 EM (Z-3D): CNN が元から puyo と判定したが浮遊している cell も
        # 物理的に不可能なので強制 EM 化。CellRecoveryRefiner の補正対象外でも適用。
        forced = self._force_em_airborne(out)

        # empty_in_stack 補正 (Z-3D): 列内で puyo の上下に EM 穴があれば、
        # 周囲から色を推論して puyo に補正。
        filled = self._fill_empty_in_stack(out, frame, region)
        em_count += filled

        return out, RefineStats(
            em_recovered=em_count,
            ojm_recovered=ojm_count,
            hsv_voted=vote_count,
        )

    @staticmethod
    def _revert_airborne(board: Board, original: Board) -> int:
        """補正で puyo になった cell が浮遊している場合、元値に戻す。

        Returns:
            取り消した cell 数。
        """
        reverted = 0
        for vrow in range(11):  # 0..10 (最下段=11 は対象外)
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                color = int(board.get(row, col))
                if color in (COLOR_EMPTY, COLOR_UNKNOWN):
                    continue
                below = int(board.get(row + 1, col))
                if below == COLOR_EMPTY:
                    orig_color = int(original.get(row, col))
                    if orig_color != color:
                        board.set(row, col, orig_color)
                        reverted += 1
        return reverted

    @staticmethod
    def _force_em_airborne(board: Board) -> int:
        """補正対象外の cell も含め、airborne な puyo を強制 EM 化 (Z-3D)。

        重力下で puyo が浮遊しているのは物理的に不可能なので、CNN が
        元々 puyo と判定していても EM に補正する。下から順に走査し、
        既に EM になった cell の上の puyo も連鎖的に EM 化する。

        Z-3K で snapshot 化を試行したが v13 で悪化 (89.5%)、cascade が
        column 単位の CNN 誤検出 hide に有効と判明 → 元実装維持。

        Returns:
            強制 EM 化した cell 数。
        """
        forced = 0
        # 下から上へ走査 (10..0)
        for vrow in range(10, -1, -1):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                color = int(board.get(row, col))
                if color in (COLOR_EMPTY, COLOR_UNKNOWN):
                    continue
                below = int(board.get(row + 1, col))
                if below == COLOR_EMPTY:
                    board.set(row, col, COLOR_EMPTY)
                    forced += 1
        return forced

    def _fill_empty_in_stack(
        self, board: Board, frame: np.ndarray, region,
    ) -> int:
        """列内で EM の穴を周囲色から推論して埋める (Z-3D)。

        対象: 列の最上端 puyo 行 (top_filled) より下にある EM cell。
        補正方針:
            1. その cell の HSV を再評価し、puyo っぽければ HSV 主要色を採用
            2. HSV が EM/?? を返すなら EM 維持 (補正しない、誤補正リスク回避)

        Returns:
            補正した cell 数。
        """
        filled = 0
        for col in range(BOARD_COLS):
            # 列の最上端 puyo 行を探す
            top_filled = -1
            for vrow in range(12):
                row = vrow + HIDDEN_ROWS
                color = int(board.get(row, col))
                if color not in (COLOR_EMPTY, COLOR_UNKNOWN):
                    top_filled = row
                    break
            if top_filled < 0:
                continue
            # top_filled より下で EM の cell を埋める
            for row in range(top_filled + 1, top_filled + 12):
                if row >= top_filled + 12 or row > HIDDEN_ROWS + 11:
                    break
                if int(board.get(row, col)) != COLOR_EMPTY:
                    continue
                patch = self._extract_inner(frame, region, row, col)
                if patch is None or patch.size == 0:
                    continue
                s_mean, v_mean = self._hsv_mean(patch)
                # 高彩度 → HSV 主要色
                if s_mean >= self._em_s_min and v_mean >= self._em_v_min:
                    hsv_color = self._classify_hsv(patch)
                    if hsv_color not in (COLOR_EMPTY, COLOR_UNKNOWN):
                        board.set(row, col, hsv_color)
                        filled += 1
                        continue
                # OJM 範囲なら OJM
                if (self._enable_ojm_recovery
                        and s_mean < self._ojm_s_max
                        and self._ojm_v_min <= v_mean <= self._ojm_v_max):
                    board.set(row, col, COLOR_OJAMA)
                    filled += 1
        return filled

    def _recover_em(
        self, patch: np.ndarray, s_mean: float, v_mean: float,
    ) -> int:
        """EM/UNKNOWN 判定の cell を、HSV 解析で puyo に救う。

        - S が高ければ HSV 主要色 (戦略 1)
        - 低彩度 + 中域 V なら OJM (戦略 2)
        - どちらでもなければ EM のまま
        """
        if s_mean >= self._em_s_min and v_mean >= self._em_v_min:
            hsv_color = self._classify_hsv(patch)
            if hsv_color not in (COLOR_EMPTY, COLOR_UNKNOWN):
                return hsv_color
            # HSV も EM/?? と判定 → puyo として復元できないので EM
            return COLOR_EMPTY
        if (self._enable_ojm_recovery
                and s_mean < self._ojm_s_max
                and self._ojm_v_min <= v_mean <= self._ojm_v_max):
            return COLOR_OJAMA
        return COLOR_EMPTY

    def _classify_hsv(self, patch: np.ndarray) -> int:
        """HSV classifier で patch を分類。"""
        if self._hsv is None:
            return COLOR_EMPTY
        try:
            return int(self._hsv.classify(patch))
        except Exception:
            return COLOR_EMPTY

    def calibrate_thresholds(
        self,
        bg_frames: list[np.ndarray],
        regions: list,
    ) -> None:
        """Z-3H: 試合開始 BG frame の HSV 統計から動画別閾値を算出。

        試合開始前の BG frame は全 EM 想定。各 cell の S/V 値を集めて
        統計を取り、「真の puyo はこれより顕著に高い S」になるよう
        EM_S_MIN を自動設定。低彩度 BG 動画でも適切な閾値が得られる。

        Args:
            bg_frames: 試合開始前後の 7 frame など (1080p)。
            regions: 評価する [DEFAULT_P1_REGION, DEFAULT_P2_REGION]。
        """
        if not bg_frames:
            return
        from src.image_reader import (  # 遅延 import で循環回避
            DEFAULT_P1_REGION, DEFAULT_P2_REGION,
        )
        sat_values: list[float] = []
        val_values: list[float] = []
        for frame in bg_frames:
            for region in regions:
                for vrow in range(12):
                    row = vrow + HIDDEN_ROWS
                    for col in range(BOARD_COLS):
                        patch = self._extract_inner(
                            frame, region, row, col,
                        )
                        if patch is None or patch.size == 0:
                            continue
                        s, v = self._hsv_mean(patch)
                        sat_values.append(s)
                        val_values.append(v)
        if not sat_values:
            return
        s_mean = float(np.mean(sat_values))
        s_std = float(np.std(sat_values))
        # EM_S_MIN: 真 EM 平均 + 1σ、default 以上、上限 90
        # 高彩度 BG でも閾値が上がりすぎないよう上限制約 (低彩度 puyo の救済維持)
        candidate = int(s_mean + s_std)
        self._em_s_min = max(EM_RECOVERY_S_MIN, min(candidate, 90))
        # OJM 範囲は既存 default 維持 (動画別 BG での OJM 統計取得は無理)
        self._calibrated = True

    @staticmethod
    def _hsv_mean(patch: np.ndarray) -> tuple[float, float]:
        """patch の HSV (S, V) 平均。"""
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 1])), float(np.mean(hsv[:, :, 2]))

    @staticmethod
    def _hsv_mean_from_full(
        hsv_full: np.ndarray, region, row: int, col: int,
    ) -> tuple[float, float]:
        """Z-3C: 事前計算した hsv_full から該当 cell の S/V 平均を直接抽出。

        BGR patch → cvtColor を避けて高速化。
        """
        h, w = hsv_full.shape[:2]
        x1, y1, x2, y2 = region.cell_sample_rect(row, col)
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        if y2 <= y1 or x2 <= x1:
            return 0.0, 0.0
        slice_ = hsv_full[y1:y2, x1:x2]
        return float(np.mean(slice_[:, :, 1])), float(np.mean(slice_[:, :, 2]))

    @staticmethod
    def _extract_inner(
        frame: np.ndarray, region, row: int, col: int,
    ) -> np.ndarray | None:
        """region.cell_sample_rect の patch をそのまま返す。

        中央 50% に絞ると puyo 中心の光沢で色相がずれ、HSV.classify と
        標準 ImageReader の結果が乖離する (Phase Z-2 で実証)。
        cell_sample_rect は既に CELL_SAMPLE_RATIO=0.5 で中央 50% を
        切り出しているので、それ以上絞らない。
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region.cell_sample_rect(row, col)
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        return patch


__all__ = [
    "CellRecoveryRefiner",
    "RefineStats",
    "EM_RECOVERY_S_MIN",
    "EM_RECOVERY_V_MIN",
    "OJM_RECOVERY_S_MAX",
    "OJM_RECOVERY_V_MIN",
    "OJM_RECOVERY_V_MAX",
    "HSV_VOTE_S_MIN",
]
