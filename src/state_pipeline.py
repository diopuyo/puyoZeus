"""W1.1: フィールド/ネクスト/得点/お邪魔を 1 フレームから一括抽出する State パイプライン。

強化学習エージェントの入力 (state) として必要な 4 項目を単一 API で
取得する薄い統合ラッパー。各サブシステム (ImageReader, NextDetector,
ScoreOcr, OjamaScoreInferrer 等) はすでに実装済み。

API:
    pipeline = StatePipeline()
    state = pipeline.extract(frame_bgr, t_sec)
    # state.board_p1, state.board_p2, state.next_p1, state.score_p1, ...

設計方針:
    - **状態は内部に保持する** (フレーム間で score 差分・leftover を計算)
    - reset() で試合開始時の初期化
    - お邪魔ぷよ推論は第一バージョンでは「score 差分 → 累積」のみ
      (視覚予告クロスチェックは Phase W2 で追加予定)
    - EnhancedBoardTracker (V2.4) と統合 → next_pair で色補正自動適用
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.board import Board
from src.ojama_accounting import OjamaAccountSnapshot
from src.probabilistic_board import ProbabilisticBoard


@dataclass(frozen=True)
class GameState:
    """1 フレームから抽出した完全な状況情報。"""
    t_sec: float
    # フィールド (確定盤面、隠し段は UNKNOWN or 推論結果の最尤色)
    board_p1: Board
    board_p2: Board
    # ネクスト・ダブルネクスト
    next_p1: tuple[int, int] | None  # (top, bot)
    next_p2: tuple[int, int] | None
    dnext_p1: tuple[int, int] | None
    dnext_p2: tuple[int, int] | None
    # 得点
    score_p1: int | None
    score_p2: int | None
    score_confidence_p1: float
    score_confidence_p2: float
    # 予告お邪魔 (1P 側に降ってくる予告お邪魔 = 2P が発火した連鎖が source)
    pending_ojama_p1: int
    pending_ojama_p2: int
    # 状態フラグ
    is_match_end_locked: bool
    is_telop_visible: bool
    # W3.0: 量子的盤面 (隠し段+お邪魔の確率分布込み)。None なら未推論。
    pboard_p1: ProbabilisticBoard | None = None
    pboard_p2: ProbabilisticBoard | None = None
    # Ojama Accounting MVP: 5 帳簿スナップショット (optional、backwards compat)
    ojama_snapshot: OjamaAccountSnapshot | None = None


class StatePipeline:
    """4 項目を 1 フレームから一括抽出する State 抽出器。"""

    # OCR 信頼度がこの値未満なら score 観測を無視 (連鎖アニメ中で表示不安定)
    SCORE_CONF_THRESHOLD: float = 0.50
    # 1 step で許容するスコア増加上限 (これを超えるなら連鎖アニメ途中の不安定値とみなしスキップ)
    SCORE_DELTA_MAX: int = 50000

    def __init__(
        self,
        cnn_model_path: str | Path = "models/cnn_phase_u_v16.pt",
        use_enhanced_tracker: bool = True,
        use_match_end_detector: bool = True,
        use_telop_detector: bool = True,
        ojama_rate_base: int = 70,
        score_conf_threshold: float = SCORE_CONF_THRESHOLD,
        score_delta_max: int = SCORE_DELTA_MAX,
        use_bg_empty_detector: bool = False,
        use_score_physics_refiner: bool = False,
        use_per_video_calibrator: bool = True,
        use_temporal_voting: bool = True,
        temporal_voting_window: int = 3,
        use_score_eraser: bool = True,
        use_pair_landing_check: bool = True,
        use_chain_animation_detector: bool = False,
        use_puyo_stability: bool = False,
        use_cell_recovery: bool = True,
        use_online_hsv: bool = False,
        use_cell_anomaly: bool = False,
        use_hsv_anomaly: bool = False,
        use_connectivity_outlier: bool = False,
        use_stability: bool = False,
    ) -> None:
        self._score_conf_threshold = float(score_conf_threshold)
        self._score_delta_max = int(score_delta_max)
        self._init_image_reader(cnn_model_path)
        self._init_next_detector()
        self._init_score_ocr()
        self._init_aux(
            use_match_end_detector=use_match_end_detector,
            use_telop_detector=use_telop_detector,
        )
        self._init_ojama(rate_base=ojama_rate_base)
        self._init_trackers(use_enhanced_tracker)
        # W10-D: 試合前盤面ベース EM 判定 (per-cell L2 距離)
        if use_bg_empty_detector:
            from src.bg_empty_detector import BgEmptyDetector
            # threshold は厳しめ (false positive 抑制) — strict mode
            self._bg_em: "BgEmptyDetector | None" = BgEmptyDetector(
                threshold=10.0,
            )
        else:
            self._bg_em = None
        # W10-B: score 連動の物理推論補正 (temporal EM persistence)
        if use_score_physics_refiner:
            from src.score_physics_refiner import ScorePhysicsRefiner
            self._score_physics_p1: "ScorePhysicsRefiner | None" = (
                ScorePhysicsRefiner()
            )
            self._score_physics_p2: "ScorePhysicsRefiner | None" = (
                ScorePhysicsRefiner()
            )
        else:
            self._score_physics_p1 = None
            self._score_physics_p2 = None
        # W11-C: Per-video 色キャリブレーション (BGR shift)
        if use_per_video_calibrator:
            from src.per_video_calibrator import PerVideoCalibrator
            self._calibrator: "PerVideoCalibrator | None" = (
                PerVideoCalibrator()
            )
        else:
            self._calibrator = None
        # W11-D: 過去 N フレーム多数決
        if use_temporal_voting:
            from src.temporal_voting_refiner import TemporalVotingRefiner
            import os
            tv_window = int(os.environ.get(
                "PHASE_Z_TV_WINDOW", temporal_voting_window,
            ))
            self._temporal_p1: "TemporalVotingRefiner | None" = (
                TemporalVotingRefiner(window=tv_window)
            )
            self._temporal_p2: "TemporalVotingRefiner | None" = (
                TemporalVotingRefiner(window=tv_window)
            )
        else:
            self._temporal_p1 = None
            self._temporal_p2 = None
        # W12-B: score 連動 4+ cluster 強制 EM
        if use_score_eraser:
            from src.score_eraser import ScoreBasedEraser
            self._score_eraser: "ScoreBasedEraser | None" = (
                ScoreBasedEraser()
            )
        else:
            self._score_eraser = None
        # W13-A: pair landing constraint
        if use_pair_landing_check:
            from src.pair_landing_check import PairLandingCheck
            self._pair_landing: "PairLandingCheck | None" = (
                PairLandingCheck()
            )
        else:
            self._pair_landing = None
        # W14-C: chain animation detector
        if use_chain_animation_detector:
            from src.chain_animation_detector import (
                ChainAnimationDetector,
            )
            self._chain_anim: "ChainAnimationDetector | None" = (
                ChainAnimationDetector()
            )
        else:
            self._chain_anim = None
        # W15-A: puyo stability refiner
        if use_puyo_stability:
            from src.puyo_stability_refiner import PuyoStabilityRefiner
            self._stability: "PuyoStabilityRefiner | None" = (
                PuyoStabilityRefiner()
            )
        else:
            self._stability = None
        # Z-2: cell 単位の検出漏れ・色誤認補正 (HSV ベース)
        if use_cell_recovery:
            from src.cell_recovery_refiner import (
                CellRecoveryRefiner,
                EM_RECOVERY_S_MIN, EM_RECOVERY_V_MIN,
                HSV_VOTE_S_MIN,
            )
            from src.chain_phase_detector import ChainPhaseDetector
            from src.next_linked_refiner import NextLinkedColorRefiner
            classifier = getattr(self._image_reader, "_classifier", None)
            hsv_classifier = (
                getattr(classifier, "_hsv", None) if classifier else None
            )
            # threshold は env var で override 可能 (試行 J 用)
            import os
            em_s = int(os.environ.get(
                "PHASE_Z_EM_S_MIN", EM_RECOVERY_S_MIN,
            ))
            em_v = int(os.environ.get(
                "PHASE_Z_EM_V_MIN", EM_RECOVERY_V_MIN,
            ))
            vote_s = int(os.environ.get(
                "PHASE_Z_HSV_VOTE_S_MIN", HSV_VOTE_S_MIN,
            ))
            self._cell_recovery: "CellRecoveryRefiner | None" = (
                CellRecoveryRefiner(
                    hsv_classifier,
                    em_s_min=em_s,
                    em_v_min=em_v,
                    hsv_vote_s_min=vote_s,
                    enable_ojm_recovery=True,
                )
                if hsv_classifier is not None else None
            )
            # Z-3E: CellRecoveryRefiner 後段の next_pair 連動補正
            self._next_linked: "NextLinkedColorRefiner | None" = (
                NextLinkedColorRefiner()
            )
            # Z-3G: 連鎖中フェーズ判定 (CellRecoveryRefiner の物理違反補正を制御)
            self._chain_phase: "ChainPhaseDetector | None" = (
                ChainPhaseDetector()
            )
        else:
            self._cell_recovery = None
            self._next_linked = None
            self._chain_phase = None
        # Z-3I: 未知動画リアルタイム HSV 範囲学習
        if use_online_hsv:
            from src.online_hsv_calibrator import OnlineHsvCalibrator
            self._online_hsv: "OnlineHsvCalibrator | None" = (
                OnlineHsvCalibrator()
            )
        else:
            self._online_hsv = None
        self._hsv_injected: bool = False  # 動画別 ranges 注入済フラグ
        # Z-3J: cell hash anomaly detection (連鎖アニメ・落下中の不安定 cell 救済)
        if use_cell_anomaly:
            from src.cell_anomaly_detector import CellAnomalyDetector
            self._cell_anomaly: "CellAnomalyDetector | None" = (
                CellAnomalyDetector()
            )
        else:
            self._cell_anomaly = None
        # Z-3J': HSV mean ベースの anomaly detection (pHash 版の改良)
        if use_hsv_anomaly:
            from src.cell_hsv_anomaly_detector import CellHsvAnomalyDetector
            self._hsv_anomaly: "CellHsvAnomalyDetector | None" = (
                CellHsvAnomalyDetector()
            )
        else:
            self._hsv_anomaly = None
        # 試行 A: connectivity outlier 補正 (孤立 1 cell を周囲色に)
        if use_connectivity_outlier:
            from src.connectivity_outlier_refiner import (
                ConnectivityOutlierRefiner,
            )
            self._connectivity_outlier: (
                "ConnectivityOutlierRefiner | None"
            ) = ConnectivityOutlierRefiner()
        else:
            self._connectivity_outlier = None
        # 試行 G: cell stability tracker (HSV σ で不安定 cell を補正)
        if use_stability:
            from src.cell_stability_tracker import CellStabilityTracker
            self._stability_tracker: (
                "CellStabilityTracker | None"
            ) = CellStabilityTracker()
        else:
            self._stability_tracker = None
        self.reset()

    # ============================
    # 初期化
    # ============================

    def _init_image_reader(self, cnn_model_path: str | Path) -> None:
        from src.hybrid_classifier import HybridClassifier
        from src.image_reader import ImageReader
        classifier_obj = None
        if cnn_model_path and Path(cnn_model_path).exists():
            try:
                import torch
                state = torch.load(
                    str(cnn_model_path), map_location="cpu",
                    weights_only=True,
                )
                # v10 (16x16 ResNet) は state_dict に "conv1.weight" を含む。
                # それ以外は v7-v9 (8x8) として CnnPatchClassifier を使用。
                is_v10 = any("conv1.weight" in k for k in state.keys())
                if is_v10:
                    from src.patch_classifier_v2 import CnnPatchClassifierV2
                    cnn = CnnPatchClassifierV2()
                else:
                    from src.patch_classifier import CnnPatchClassifier
                    cnn = CnnPatchClassifier()
                cnn._model.load_state_dict(state)
                cnn._model.eval()
                # Z-3C: GPU 利用可能なら CUDA に移動
                if hasattr(cnn, "to_device"):
                    try:
                        cnn.to_device("cuda")
                    except Exception:
                        pass
                classifier_obj = HybridClassifier(cnn_classifier=cnn)
            except Exception:
                classifier_obj = None
        self._image_reader = ImageReader(
            classifier=classifier_obj,
            use_match_state=True,
            use_ui_mask=True,
            use_telop_mask=True,
        )

    def _init_next_detector(self) -> None:
        from src.next_detector import NextDetector
        from src.stable_next_detector import StableNextDetector
        try:
            base = NextDetector.load_default()
            # B4: 安定化レイヤー (連続 2 フレーム同色のみ採用、初動応答性改善)
            self._next_detector: "StableNextDetector | None" = (
                StableNextDetector(base, stability_window=2)
            )
        except Exception:
            self._next_detector = None

    def _init_score_ocr(self) -> None:
        from src.score_ocr import ScoreOcr
        try:
            self._score_ocr: "ScoreOcr | None" = ScoreOcr.load_default()
        except Exception:
            self._score_ocr = None

    def _init_aux(
        self,
        use_match_end_detector: bool,
        use_telop_detector: bool,
    ) -> None:
        if use_match_end_detector:
            from src.match_end_detector import MatchEndDetector
            self._match_end: "MatchEndDetector | None" = (
                MatchEndDetector.load_default()
            )
        else:
            self._match_end = None
        if use_telop_detector:
            from src.telop_detector import TelopDetector
            self._telop: "TelopDetector | None" = TelopDetector.load_default()
        else:
            self._telop = None

    def _init_ojama(self, rate_base: int) -> None:
        from src.ojama_score_inferrer import OjamaScoreInferrer
        # 既存 OjamaScoreInferrer は後方互換のため保持
        # OjamaAccountingTracker はここでは初期化しない。
        # (GameState.ojama_snapshot=None 固定、新 API は recognition_pipeline 側で統合)
        self._ojama_inferrer = OjamaScoreInferrer(rate=rate_base)

    def _init_trackers(self, use_enhanced_tracker: bool) -> None:
        if use_enhanced_tracker:
            from src.enhanced_board_tracker import EnhancedBoardTracker
            self._tracker_p1: "EnhancedBoardTracker | None" = (
                EnhancedBoardTracker()
            )
            self._tracker_p2: "EnhancedBoardTracker | None" = (
                EnhancedBoardTracker()
            )
        else:
            self._tracker_p1 = None
            self._tracker_p2 = None

    # ============================
    # 公開 API
    # ============================

    def set_background_fingerprints_from_video(
        self,
        cap,  # cv2.VideoCapture
        bg_fp_time_sec: float,
        n_frames: int = 7,
        offsets_sec: tuple[float, ...] = (-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5),
    ) -> bool:
        """指定動画の試合開始秒付近のフレームから BG FP を取得して設定。

        W10-D: 同時に BgEmptyDetector も同フレームでキャリブレーション。

        Returns:
            bool: 成功したら True。
        """
        import cv2
        from src.background_fingerprint import capture_pair_robust
        from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
        bg_frames = []
        for offset in offsets_sec:
            t = max(0.0, bg_fp_time_sec + offset)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fb = cap.read()
            if not ok or fb is None:
                continue
            if fb.shape[:2] != (1080, 1920):
                fb = cv2.resize(
                    fb, (1920, 1080), interpolation=cv2.INTER_AREA,
                )
            bg_frames.append(fb)
        if not bg_frames:
            return False
        p1_t = (
            DEFAULT_P1_REGION.x, DEFAULT_P1_REGION.y,
            DEFAULT_P1_REGION.width, DEFAULT_P1_REGION.height,
        )
        p2_t = (
            DEFAULT_P2_REGION.x, DEFAULT_P2_REGION.y,
            DEFAULT_P2_REGION.width, DEFAULT_P2_REGION.height,
        )
        fp1, fp2 = capture_pair_robust(bg_frames, p1_t, p2_t)
        self._image_reader.set_background_fingerprints(fp1, fp2)
        # W10-D: BgEmptyDetector もここで初期化 (per-cell 平均色)
        if self._bg_em is not None:
            self._bg_em.reset()
            for i, fb in enumerate(bg_frames):
                self._bg_em.calibrate_from_frame(
                    fb, accumulate=(i > 0),
                )
        # W11-C: PerVideoCalibrator も同フレームで calibrate
        if self._calibrator is not None:
            self._calibrator.calibrate_from_frames(bg_frames)
        # Z-3H: CellRecoveryRefiner の閾値も BG 統計で動画別調整
        if self._cell_recovery is not None:
            self._cell_recovery.calibrate_thresholds(
                bg_frames, [DEFAULT_P1_REGION, DEFAULT_P2_REGION],
            )
        return True

    def reset(self, match_start_sec: float | None = None) -> None:
        """試合開始時に呼ぶ。内部状態を初期化。"""
        self._prev_score_p1: int | None = None
        self._prev_score_p2: int | None = None
        self._leftover_p1: int = 0
        self._leftover_p2: int = 0
        self._pending_ojama_p1: int = 0
        self._pending_ojama_p2: int = 0
        self._match_start_sec: float | None = match_start_sec
        # Ojama Accounting MVP: tracker リセット
        if hasattr(self, "_ojama_accounting"):
            self._ojama_accounting.reset(match_start_sec=match_start_sec)
        if self._tracker_p1 is not None:
            self._tracker_p1.reset()
        if self._tracker_p2 is not None:
            self._tracker_p2.reset()
        if self._match_end is not None:
            self._match_end.reset()
        if self._next_detector is not None and hasattr(
            self._next_detector, "reset",
        ):
            self._next_detector.reset()
        if self._score_physics_p1 is not None:
            self._score_physics_p1.reset()
        if self._score_physics_p2 is not None:
            self._score_physics_p2.reset()
        if self._temporal_p1 is not None:
            self._temporal_p1.reset()
        if self._temporal_p2 is not None:
            self._temporal_p2.reset()
        if self._score_eraser is not None:
            self._score_eraser.reset()
        if self._pair_landing is not None:
            self._pair_landing.reset()
        if self._chain_anim is not None:
            self._chain_anim.reset()
        if self._stability is not None:
            self._stability.reset()
        if self._chain_phase is not None:
            self._chain_phase.reset()
        if self._online_hsv is not None:
            self._online_hsv.reset()
            self._hsv_injected = False
        if self._cell_anomaly is not None:
            self._cell_anomaly.reset()
        if self._hsv_anomaly is not None:
            self._hsv_anomaly.reset()
        if self._stability_tracker is not None:
            self._stability_tracker.reset()
        # W3.0: 隠し段推論用の前盤面・前 next_pair を保持
        self._prev_board_p1: Board | None = None
        self._prev_board_p2: Board | None = None
        self._prev_next_p1: tuple[int, int] | None = None
        self._prev_next_p2: tuple[int, int] | None = None
        # W3.0/B7: お邪魔位置推論用の前 pending 値
        self._prev_pending_ojama_p1: int = 0
        self._prev_pending_ojama_p2: int = 0

    def extract(
        self, frame_bgr: np.ndarray, t_sec: float,
    ) -> GameState:
        """1 フレームから完全な GameState を抽出。"""
        frame_1080 = self._ensure_1080p(frame_bgr)

        # 1. 試合終了ロックダウン (画像のみで判定)
        match_end_locked = False
        if self._match_end is not None:
            match_end_locked = self._match_end.update(frame_1080, t_sec)

        # 2. テロップ表示有無
        is_telop = False
        if self._telop is not None:
            is_telop = self._telop.is_visible(frame_1080)

        # 3. 盤面観測 → EnhancedBoardTracker で時系列フィルタ
        board_p1, board_p2 = self._extract_boards(frame_1080)

        # 4. ネクスト・ダブルネクスト
        next_p1, next_p2, dnext_p1, dnext_p2 = self._extract_next(frame_1080)

        # 5. 得点
        score_p1, score_p2, conf_p1, conf_p2 = self._extract_score(frame_1080)

        # 6. お邪魔 (score 差分 → 累積、相殺は今回未対応)
        # 信頼度低い OCR は破棄 (連鎖アニメ中の不安定 score を除外)
        score_p1_filtered = (
            score_p1 if conf_p1 >= self._score_conf_threshold else None
        )
        score_p2_filtered = (
            score_p2 if conf_p2 >= self._score_conf_threshold else None
        )
        if not match_end_locked:
            # ojama pending 計算 (旧 OjamaScoreInferrer 系の既存ロジック維持)
            # 新 OjamaAccountingTracker はここでは呼ばない。
            # GameState.ojama_snapshot の再統合は recognition_pipeline 経由で行う (申し送り)。
            self._update_ojama_pending(
                score_p1_filtered, score_p2_filtered, t_sec,
            )

        # W10-D: BgEmptyDetector で per-cell EM 強制 (試合前盤面と類似なら EM)
        if self._bg_em is not None and self._bg_em.bg_features:
            board_p1 = self._apply_bg_em(frame_1080, "1P", board_p1)
            board_p2 = self._apply_bg_em(frame_1080, "2P", board_p2)

        # 7. tracker に next_pair 反映 (時系列フィルタ用)
        if self._tracker_p1 is not None:
            board_p1 = self._tracker_p1.update(board_p1, next_pair=next_p1)
        if self._tracker_p2 is not None:
            board_p2 = self._tracker_p2.update(board_p2, next_pair=next_p2)

        # W10-B: ScorePhysicsRefiner で score 連動 EM 永続化
        if self._score_physics_p1 is not None:
            board_p1 = self._score_physics_p1.refine(
                "1P", board_p1, score_p1_filtered,
            )
        if self._score_physics_p2 is not None:
            board_p2 = self._score_physics_p2.refine(
                "2P", board_p2, score_p2_filtered,
            )
        # W11-D: 過去 N フレーム多数決 (最終出力前)
        if self._temporal_p1 is not None:
            board_p1 = self._temporal_p1.refine("1P", board_p1)
        if self._temporal_p2 is not None:
            board_p2 = self._temporal_p2.refine("2P", board_p2)
        # W12-B: score 連動 4+ cluster 強制 EM (chain animation 期間)
        if self._score_eraser is not None:
            board_p1 = self._score_eraser.refine(
                "1P", board_p1, score_p1_filtered,
            )
            board_p2 = self._score_eraser.refine(
                "2P", board_p2, score_p2_filtered,
            )
        # W13-A: pair landing constraint (新着 cell 色 ↔ next_pair)
        if self._pair_landing is not None:
            board_p1 = self._pair_landing.refine(
                "1P", board_p1, next_p1, dnext_p1,
            )
            board_p2 = self._pair_landing.refine(
                "2P", board_p2, next_p2, dnext_p2,
            )
        # W14-C: 連鎖アニメ検出 (画面 motion で前 stable に戻す)
        if self._chain_anim is not None:
            try:
                from src.image_reader import (
                    DEFAULT_P1_REGION, DEFAULT_P2_REGION,
                )
                from src.board import HIDDEN_ROWS, BOARD_COLS
                # field crop (簡易)
                x1_1, y1_1, _, _ = DEFAULT_P1_REGION.cell_sample_rect(
                    HIDDEN_ROWS, 0,
                )
                _, _, x2_1, y2_1 = DEFAULT_P1_REGION.cell_sample_rect(
                    HIDDEN_ROWS + 11, BOARD_COLS - 1,
                )
                x1_2, y1_2, _, _ = DEFAULT_P2_REGION.cell_sample_rect(
                    HIDDEN_ROWS, 0,
                )
                _, _, x2_2, y2_2 = DEFAULT_P2_REGION.cell_sample_rect(
                    HIDDEN_ROWS + 11, BOARD_COLS - 1,
                )
                p1_field = frame_1080[y1_1:y2_1, x1_1:x2_1]
                p2_field = frame_1080[y1_2:y2_2, x1_2:x2_2]
                board_p1 = self._chain_anim.refine(
                    "1P", p1_field, board_p1,
                )
                board_p2 = self._chain_anim.refine(
                    "2P", p2_field, board_p2,
                )
            except Exception:
                pass
        # W15-A: 物理拘束 (連鎖/ツモ着地以外で cell 不変)
        if self._stability is not None:
            board_p1 = self._stability.refine(
                "1P", board_p1,
                score_p1_filtered, next_p1,
                self._pending_ojama_p1,
            )
            board_p2 = self._stability.refine(
                "2P", board_p2,
                score_p2_filtered, next_p2,
                self._pending_ojama_p2,
            )

        # Z-3G: 連鎖フェーズ判定 (CellRecoveryRefiner と OnlineHsvCalibrator で共有)
        is_chain_p1 = False
        is_chain_p2 = False
        if self._chain_phase is not None:
            chain_res = self._chain_phase.update(
                t_sec, board_p1, board_p2,
                score_p1_filtered, score_p2_filtered,
            )
            is_chain_p1 = chain_res.is_chain_p1
            is_chain_p2 = chain_res.is_chain_p2
        # NOTE: OjamaAccountingTracker の新 API (on_state_transition 等) は
        # state_pipeline 層では BoardStateMachine に per-side アクセスできないため
        # ここからは呼ばない。GameState.ojama_snapshot は None のまま。
        # 新 API の統合は recognition_pipeline.py 経由で行う (申し送り参照)。
        # chain フラグを保存 (次フレーム用)
        self._last_is_chain_p1: bool = is_chain_p1
        self._last_is_chain_p2: bool = is_chain_p2

        # Z-2: CellRecoveryRefiner で最終的な検出漏れ・色誤認補正
        # 他補正レイヤー全部の後に適用 (tracker 等の影響を受けないよう最後に)
        if self._cell_recovery is not None:
            from src.image_reader import (
                DEFAULT_P1_REGION, DEFAULT_P2_REGION,
            )
            target = frame_1080
            if (self._calibrator is not None
                    and self._calibrator.n_calib_frames > 0):
                target = self._calibrator.apply(frame_1080)
            # Z-3C: HSV 全体変換を P1/P2 で共有
            hsv_full = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
            board_p1, _ = self._cell_recovery.refine(
                target, DEFAULT_P1_REGION, board_p1,
                is_chain=is_chain_p1, hsv_full=hsv_full,
            )
            board_p2, _ = self._cell_recovery.refine(
                target, DEFAULT_P2_REGION, board_p2,
                is_chain=is_chain_p2, hsv_full=hsv_full,
            )

        # Z-3E: NextLinkedColorRefiner 統合は逆効果のため無効化
        # (連鎖境界 frame で誤動作、個別 GT を悪化させた)

        # Z-3J: CellAnomalyDetector で連鎖アニメ・落下中の不安定 cell を補正
        if self._cell_anomaly is not None:
            from src.image_reader import (
                DEFAULT_P1_REGION, DEFAULT_P2_REGION,
            )
            board_p1, _ = self._cell_anomaly.refine(
                frame_1080, DEFAULT_P1_REGION, board_p1, "1P",
                is_chain=is_chain_p1,
            )
            board_p2, _ = self._cell_anomaly.refine(
                frame_1080, DEFAULT_P2_REGION, board_p2, "2P",
                is_chain=is_chain_p2,
            )
        # Z-3J': HSV mean anomaly (pHash 版の改良、puyo 自然変動を許容)
        if self._hsv_anomaly is not None:
            from src.image_reader import (
                DEFAULT_P1_REGION, DEFAULT_P2_REGION,
            )
            board_p1, _ = self._hsv_anomaly.refine(
                frame_1080, DEFAULT_P1_REGION, board_p1, "1P",
                is_chain=is_chain_p1,
            )
            board_p2, _ = self._hsv_anomaly.refine(
                frame_1080, DEFAULT_P2_REGION, board_p2, "2P",
                is_chain=is_chain_p2,
            )
        # 試行 A: connectivity outlier (孤立 1 cell 補正)
        if self._connectivity_outlier is not None:
            board_p1, _ = self._connectivity_outlier.refine(
                board_p1, is_chain=is_chain_p1,
            )
            board_p2, _ = self._connectivity_outlier.refine(
                board_p2, is_chain=is_chain_p2,
            )
        # 試行 G: cell stability tracker (HSV σ で不安定検出)
        if self._stability_tracker is not None:
            from src.image_reader import (
                DEFAULT_P1_REGION, DEFAULT_P2_REGION,
            )
            board_p1, _ = self._stability_tracker.refine(
                frame_1080, DEFAULT_P1_REGION, board_p1, "1P",
                is_chain=is_chain_p1,
            )
            board_p2, _ = self._stability_tracker.refine(
                frame_1080, DEFAULT_P2_REGION, board_p2, "2P",
                is_chain=is_chain_p2,
            )

        # Z-3I: OnlineHsvCalibrator で動画別 HSV 範囲を学習
        if self._online_hsv is not None:
            from src.image_reader import (
                DEFAULT_P1_REGION, DEFAULT_P2_REGION,
            )
            target_for_hsv = frame_1080
            if (self._calibrator is not None
                    and self._calibrator.n_calib_frames > 0):
                target_for_hsv = self._calibrator.apply(frame_1080)
            # CNN proba/HSV color grid なし → board 値のみ参照 (簡易学習)
            # 厳密な信頼性チェックは StatePipeline 外で行う (重複計算回避)
            self._online_hsv.update(
                target_for_hsv, DEFAULT_P1_REGION, board_p1,
                cnn_proba_grid=None, hsv_color_grid=None,
                is_chain=is_chain_p1,
            )
            self._online_hsv.update(
                target_for_hsv, DEFAULT_P2_REGION, board_p2,
                cnn_proba_grid=None, hsv_color_grid=None,
                is_chain=is_chain_p2,
            )
            # 動画別 ranges が ready なら ColorClassifier に注入 (1 回のみ)
            if not self._hsv_injected and self._online_hsv.is_ready():
                ranges = self._online_hsv.get_per_video_ranges()
                classifier = getattr(
                    self._image_reader, "_classifier", None,
                )
                hsv_classifier = (
                    getattr(classifier, "_hsv", None)
                    if classifier else None
                )
                if hsv_classifier is not None and hasattr(
                    hsv_classifier, "set_color_ranges_from_simple",
                ):
                    hsv_classifier.set_color_ranges_from_simple(ranges)
                    self._hsv_injected = True

        # W3.0: 確率的盤面 (隠し段推論 + B7 お邪魔位置推論)
        # expected_ojama: prev pending → cur pending で減少した分が「落下した ojama」
        # ただし pending は score 差分で増えるだけなので、簡易的に 「prev > 0 なら expected」 とする
        pboard_p1 = self._infer_probabilistic_full(
            self._prev_board_p1, board_p1, self._prev_next_p1,
            expected_ojama=self._prev_pending_ojama_p1,
        )
        pboard_p2 = self._infer_probabilistic_full(
            self._prev_board_p2, board_p2, self._prev_next_p2,
            expected_ojama=self._prev_pending_ojama_p2,
        )
        # 次フレーム用に prev 保存
        self._prev_board_p1 = board_p1.copy()
        self._prev_board_p2 = board_p2.copy()
        self._prev_next_p1 = next_p1
        self._prev_next_p2 = next_p2
        self._prev_pending_ojama_p1 = self._pending_ojama_p1
        self._prev_pending_ojama_p2 = self._pending_ojama_p2

        return GameState(
            t_sec=t_sec,
            board_p1=board_p1,
            board_p2=board_p2,
            next_p1=next_p1,
            next_p2=next_p2,
            dnext_p1=dnext_p1,
            dnext_p2=dnext_p2,
            score_p1=score_p1,
            score_p2=score_p2,
            score_confidence_p1=conf_p1,
            score_confidence_p2=conf_p2,
            pending_ojama_p1=self._pending_ojama_p1,
            pending_ojama_p2=self._pending_ojama_p2,
            is_match_end_locked=match_end_locked,
            is_telop_visible=is_telop,
            pboard_p1=pboard_p1,
            pboard_p2=pboard_p2,
            ojama_snapshot=getattr(self, "_ojama_snapshot", None),
        )

    @staticmethod
    def _infer_probabilistic_full(
        prev: Board | None,
        cur: Board,
        prev_next_pair: tuple[int, int] | None,
        expected_ojama: int = 0,
    ) -> ProbabilisticBoard:
        """W3.0 + B7: 前盤面 + next_pair + 予告 ojama から隠し段確率分布を推論。

        1. hidden_row_inferrer で隠し段のぷよ位置確率
        2. expected_ojama > 0 なら ojama_position_inferrer を追加適用
           (隠し段 OJM 確率を上書き)
        """
        if prev is None:
            return ProbabilisticBoard.from_board(cur)
        from src.hidden_row_inferrer import infer_hidden_row
        from src.ojama_position_inferrer import infer_ojama_positions
        from src.board import COLOR_OJAMA
        # 1. 隠し段ぷよ位置推論
        if prev_next_pair is not None:
            pboard, _ = infer_hidden_row(prev, cur, prev_next_pair)
        else:
            pboard = ProbabilisticBoard.from_board(cur)
        # 2. お邪魔位置推論 (隠し段の OJM 確率を上書き)
        if expected_ojama > 0:
            ojama_pb, ojama_res = infer_ojama_positions(
                prev, cur, expected_ojama,
            )
            # OJM 候補列のみ pboard に反映 (hidden_row 推論より優先)
            for col in ojama_res.candidate_cols:
                ojama_cell = ojama_pb.cell(0, col)
                ojama_p = ojama_cell.get(COLOR_OJAMA)
                if ojama_p > 0:
                    # 既存 pboard の隠し段に OJM 確率を統合
                    cur_cell = pboard.cell(0, col)
                    new_probs = dict(cur_cell.probs)
                    # OJM 確率を上書き、他色は残量で正規化
                    new_probs[COLOR_OJAMA] = ojama_p
                    pboard.set_distribution(0, col, new_probs)
        return pboard

    # 互換用の旧名 (テスト互換)
    _infer_probabilistic = _infer_probabilistic_full

    # ============================
    # 内部抽出ヘルパ
    # ============================

    @staticmethod
    def _ensure_1080p(frame: np.ndarray) -> np.ndarray:
        if frame.shape[:2] != (1080, 1920):
            return cv2.resize(
                frame, (1920, 1080), interpolation=cv2.INTER_AREA,
            )
        return frame

    def _extract_boards(self, frame: np.ndarray) -> tuple[Board, Board]:
        try:
            # W11-C: per-video calibration を board 抽出のみに適用
            # (score OCR/next detector には影響なし)
            target_frame = frame
            if (
                self._calibrator is not None
                and self._calibrator.n_calib_frames > 0
            ):
                target_frame = self._calibrator.apply(frame)
            return self._image_reader.read_both_boards(target_frame)
        except Exception:
            return Board(), Board()

    def _apply_bg_em(
        self, frame: np.ndarray, side: str, board: Board,
    ) -> Board:
        """W10-D: 試合前盤面との類似度で EM 強制。

        非 EM 判定の cell について試合前 BG パターンと比較。
        距離が threshold 未満なら EM に上書き。
        OJAMA は降ってくるので除外。
        """
        from src.bg_empty_detector import _extract_patch
        from src.image_reader import (
            DEFAULT_P1_REGION, DEFAULT_P2_REGION,
        )
        from src.board import (
            BOARD_COLS, COLOR_EMPTY, COLOR_OJAMA,
            COLOR_UNKNOWN, HIDDEN_ROWS,
        )
        if self._bg_em is None or not self._bg_em.bg_features:
            return board
        region = (
            DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
        )
        out = board.copy()
        for vrow in range(12):
            row = vrow + HIDDEN_ROWS
            for col in range(BOARD_COLS):
                color = int(out.get(row, col))
                if color in (COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN):
                    continue
                patch = _extract_patch(frame, region, row, col)
                if patch.size == 0:
                    continue
                if self._bg_em.is_empty(side, vrow, col, patch):
                    out.set(row, col, COLOR_EMPTY)
        return out

    def _extract_next(
        self, frame: np.ndarray,
    ) -> tuple[
        tuple[int, int] | None,
        tuple[int, int] | None,
        tuple[int, int] | None,
        tuple[int, int] | None,
    ]:
        if self._next_detector is None:
            return None, None, None, None
        try:
            # B4: StableNextDetector は 3 連続同色のみ採用 (None 可)
            stable = self._next_detector.detect_both(frame)
            return (
                stable.p1_next, stable.p2_next,
                stable.p1_dnext, stable.p2_dnext,
            )
        except Exception:
            return None, None, None, None

    def _extract_score(
        self, frame: np.ndarray,
    ) -> tuple[int | None, int | None, float, float]:
        if self._score_ocr is None:
            return None, None, 0.0, 0.0
        try:
            r = self._score_ocr.read(frame)
            return (
                r.score_1p, r.score_2p,
                r.confidence_1p, r.confidence_2p,
            )
        except Exception:
            return None, None, 0.0, 0.0

    def _update_ojama_pending(
        self,
        score_p1: int | None,
        score_p2: int | None,
        t_sec: float,
    ) -> None:
        """score 差分から予告 ojama を累積し、Ojama Accounting tracker の生成処理も実行する。

        Step1.5-③: chain 相殺は extract() 末尾で update_accounting_with_chain() が担う。
        既存の _pending_ojama_p1/p2 は後方互換のため維持しつつ、
        OjamaAccountingTracker でも並行して 5 帳簿管理 (生成のみ) を行う。
        """
        elapsed = self._elapsed_sec(t_sec)
        if score_p1 is not None and self._prev_score_p1 is not None:
            delta_p1 = score_p1 - self._prev_score_p1
            if 0 < delta_p1 <= self._score_delta_max:
                pred, leftover = self._ojama_inferrer.infer_from_score_delta(
                    score_before=self._prev_score_p1,
                    score_after=score_p1,
                    fired_by="1P",
                    match_elapsed_sec=elapsed,
                    prev_leftover_sender=self._leftover_p1,
                )
                # 1P が fire → 2P pending に加算 (既存ロジック維持)
                self._pending_ojama_p2 += pred.pending
                self._leftover_p1 = leftover
        if score_p2 is not None and self._prev_score_p2 is not None:
            delta_p2 = score_p2 - self._prev_score_p2
            if 0 < delta_p2 <= self._score_delta_max:
                pred, leftover = self._ojama_inferrer.infer_from_score_delta(
                    score_before=self._prev_score_p2,
                    score_after=score_p2,
                    fired_by="2P",
                    match_elapsed_sec=elapsed,
                    prev_leftover_sender=self._leftover_p2,
                )
                self._pending_ojama_p1 += pred.pending
                self._leftover_p2 = leftover
        # 観測値を保存 (前回値として)
        if score_p1 is not None:
            self._prev_score_p1 = score_p1
        if score_p2 is not None:
            self._prev_score_p2 = score_p2

        # NOTE: OjamaAccountingTracker の旧 API (update_from_score) はここから呼ばない。
        # GameState.ojama_snapshot = None 固定。新 API は recognition_pipeline 側で統合。

    def _elapsed_sec(self, t_sec: float) -> float:
        if self._match_start_sec is None:
            return 0.0
        return max(0.0, t_sec - self._match_start_sec)


__all__ = [
    "GameState",
    "StatePipeline",
]
