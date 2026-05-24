# 画像認識ロジック全体説明資料 (2026-05-08)

> **目的**: ユーザーが Phase I で「STABLE 確定盤面 99.99% 認識」を目指すにあたり、
> 現在の認識パイプラインを系統的に理解し、改善ポイントを特定するための一次資料。
>
> 関連メモリ: `feedback_recognition_target_995.md` (99.99% 目標),
> `project_recognition_strategy_pivot.md` (state machine 主軸 pivot, 2026-05-03),
> `feedback_chain_phase_physics_only.md` (アクション中は全て物理推論),
> `project_unknown_video_realtime_hsv.md` (online HSV 三段階).

---

## 目次

1. 概要
2. 各モジュールの役割
3. 全体パイプラインのフロー詳細
4. State Machine (BoardState) 遷移詳細
5. 時系列ノイズ除去メカニズム
6. 学習で達成した認識精度
7. 既知の認識弱点
8. Phase I 改善ロードマップ (提案)
9. 参考資料 / 関連コード

---

## 1. 概要

### 1.1 入出力

| 項目 | 内容 |
|---|---|
| **入力** | 1920×1080 BGR `numpy.ndarray` フレーム (動画) + frame_idx + time_sec |
| **中間** | 1P/2P `Board` (CNN 直結、HSV+CNN ハイブリッド、UI mask 適用済み) |
| **出力** | `PipelineResult` (1P/2P 各サイドが `SideResult`) |

`SideResult` の主な属性 (`src/recognition_pipeline.py:76-97`):

- `state: BoardState` — STABLE / TSUMO_FALL / CHAIN / OJAMA_FALL / EFFECT / MENU
- `cnn_board: Board` — ImageReader 直の出力 (drift detector 入力)
- `inferred_board: Board | None` — 推論盤面 (state ごとに生成)
- `confirmed_board: Board | None` — state machine が確定した STABLE 盤面 (★ 公式真値 ★)
- `prob_board: ProbabilisticBoard | None` — STABLE 確定時の隠し段確率分布
- `drift: DriftResult` — 推論盤面と CNN 盤面の乖離量
- `score: int | None` — Score OCR 値
- `chain_event: ChainEvent | None` — 連鎖発火イベント

### 1.2 全体フロー (テキスト図)

```
            ┌──────────────────────────────────────────────────────┐
            │                  RecognitionPipeline                 │
            │                                                      │
[1080p frame] ──► MatchStateDetector  (HSV V 平均で試合中/メニュー)  │
            │       │                                              │
            │       ├──► hysteresis (10 frame), score>0 補強         │
            │       │                                              │
            │       ▼                                              │
            │   ImageReader.read_both_boards                       │
            │       │  ┌─ ROI 切出し (P1 384x720, P2 384x720)        │
            │       │  ├─ BG fingerprint (空 cell 早期判定)         │
            │       │  ├─ HybridClassifier (HSV + CNN + UI mask)     │
            │       │  ├─ ColorClassifier (HSV ルール)              │
            │       │  ├─ CnnPatchClassifier (BGR+HSV 6ch CNN)       │
            │       │  ├─ board_rules.clear_floating_above_gap      │
            │       │  └─ _infer_hidden_rows (空列 → 隠し段空)        │
            │       ▼                                              │
            │   cnn_1p_raw, cnn_2p_raw  (1P/2P Board)               │
            │       │                                              │
            │       ▼  temporal_smoothing (default N=1, no-op)      │
            │   cnn_1p, cnn_2p                                     │
            │       │                                              │
            │       ▼                                              │
            │   VideoChainTracker × 2  (puyo 数急減で連鎖検出)         │
            │   ScoreTracker × 2       (score OCR + 差分)            │
            │   NextDetector           (next/dnext pair)            │
            │       │                                              │
            │       ▼ (per side)                                   │
            │   BoardStateMachine.update(signals)                  │
            │       │  ┌─ ChainPhaseDetector  (event あり?)          │
            │       │  ├─ EffectPhaseDetector (skeleton)            │
            │       │  ├─ OjamaPhaseDetector  (相手 score 増分?)      │
            │       │  └─ TsumoPhaseDetector  (puyo +1〜+2 連続?)     │
            │       │                                              │
            │       ▼ TSUMO→STABLE 復帰時:                          │
            │   _compute_landing_inferred (色はネクスト履歴)         │
            │   infer_hidden_row (隠し段量子推論)                   │
            │   _inject_pseudo_chain_event (連鎖発火物理推論)       │
            │       │                                              │
            │       ▼ CHAIN→STABLE 復帰時:                          │
            │   ChainSimulator.simulate.final_board で上書き        │
            │       │                                              │
            │       ▼                                              │
            │   InferenceBoardGenerator.generate                   │
            │       │                                              │
            │       ▼                                              │
            │   DriftDetector.update (inferred vs cnn)             │
            │       │  needs_resync → state machine reset          │
            │       ▼                                              │
            │   SideResult (state, confirmed_board, prob_board, …) │
            └──────────────────────────────────────────────────────┘
```

### 1.3 設計原則 (現方針)

1. **STABLE state のみ CNN 認識を盤面確定に使う** (`recognition_strategy_pivot`)
2. **アクション中 (TSUMO/CHAIN/OJAMA/EFFECT) は全部物理推論** (`chain_phase_physics_only`)
3. **連続多数決 + drift detection + state machine で 99.99% を狙う** (`recognition_target_995`)

---

## 2. 各モジュールの役割

### 2.1 統合 pipeline

#### `src/recognition_pipeline.py` — RecognitionPipeline / SideResult / PipelineResult

B-1 〜 B-6 を連結する最上位 pipeline。1 frame 投入で 1P/2P 両側の `SideResult` を返す。
内部に 1P/2P 別 `BoardStateMachine`, `InferenceBoardGenerator`, `DriftDetector`,
`VideoChainTracker`, `ScoreTracker` を保持。`load_default()` で HybridClassifier
(HSV+CNN, override_prob=0.90) を組み立てる。

主な役割:

- **試合状態 hysteresis** (`MATCH_ACTIVE_HOLD_FRAMES=10` frame): 1 frame 単発の
  メニュー誤判定吸収。score>0 観測歴での補強もあり。
- **CHAIN ban** (`CHAIN_BAN_FRAMES_AFTER_MATCH_START=30` frame): 試合開始直後の
  連鎖誤検出 ban。
- **背景 FP 自動採取** (試合開始 +5 frame 以降、puyo 0 個盤面 5 枚で
  `capture_robust_fingerprint`)。
- **TSUMO_FALL → STABLE 復帰時**: `_compute_landing_inferred` で **位置は CNN 差分,
  色はネクスト履歴 (落下中ツモ色)** で確定 + `infer_hidden_row` で隠し段量子推論
  + `_inject_pseudo_chain_event` で連鎖発火物理推論。
- **CHAIN → STABLE 復帰時**: `ChainSimulator.simulate.final_board` で confirmed 上書き
  (CNN 残光・エフェクトを排除)。
- **W-α**: STABLE 確定時に `prob_board` を publish (隠し段推論結果がなければ
  `ProbabilisticBoard.from_board(confirmed)` でフォールバック)。

テスト: `tests/test_recognition_pipeline.py` (6+ tests)。

#### `src/board_recognition_pipeline.py` — BoardRecognitionPipeline (旧 Phase U)

連続フレーム前処理用の旧 pipeline。
`AnimationFilter → ImageReader → TemporalSmoother → StatefulBoardTracker →
AdaptiveBackgroundFingerprint`。Phase B 以降の `RecognitionPipeline` で役割が
吸収されており、Phase B 以降の主軸ではない。

### 2.2 単一フレーム認識

#### `src/image_reader.py` — ImageReader / ColorClassifier / BoardRegion / HsvRange

frame → Board 変換の中核。

- `BoardRegion`: 盤面 ROI (P1: x=282, y=160, w=384, h=720; P2: x=1258, y=160, w=384, h=720)。
- `ColorClassifier.classify`: HSV 中央値ルールで 7 クラス分類
  (EMPTY/RED/BLUE/GREEN/YELLOW/PURPLE/OJAMA)。`DEFAULT_COLOR_RANGES` (5 色 + ojama)。
  赤 H 11-18 拡張範囲は BGR の R-G 差で黄と区別 (`RED_GREEN_DIFF_FOR_RED=80`)。
- `ImageReader.read_board`: 各 cell (12 visible + 1 hidden = 13 行 × 6 列) で
  中央 50% (`CELL_SAMPLE_RATIO=0.5`) サンプル → batch classify → UI mask →
  `clear_floating_above_gap (min_gap=2)` → `_infer_hidden_rows` (重力推論)。
- `read_both_boards`: 1080p 強制リサイズ + テロップ検出 + ROI offset 補正対応。
- `set_color_ranges_from_simple`: Online HSV calibrator が動画別 HSV 範囲を注入する API。

テスト: `tests/test_image_reader.py` (4), `tests/test_image_reader_golden.py`。

#### `src/patch_classifier.py` — CnnPatchClassifier / GatedCnnClassifier / HsvPatchClassifier / MlpPatchClassifier / PuyoPresenceGate

セルパッチ単位の色分類器群。`PatchClassifier` 抽象基底。

- `CnnPatchClassifier`: 8×8 にリサイズし **BGR 3ch + HSV 3ch = 6ch** で入力する
  小型 CNN (Conv16-Conv16-AdaptivePool2x2-Linear32-Linear7)。GPU 対応 (`to_device`)。
  `predict_proba_batch` で多 patch 一括推論 (5-20 倍速)。
- `GatedCnnClassifier`: `PuyoPresenceGate` (puyo 存在検出: 目+陰影+彩度 +
  特定セルでは strict 2 眼ペア) → CNN 色分類 → HSV strong color override (赤 H 0-10
  / H 165-180、紫 H 125-160) → UI mask。
- `HsvPatchClassifier`: ColorClassifier の薄いラッパ。
- `MlpPatchClassifier`: numpy 純実装 MLP (テスト・比較用)。
- `PuyoPresenceGate`: 中央 12% 余白を取った領域で `_has_eyes` (暗点 4-連結 1+ 個),
  `_has_shading` (V std ≥ 12), `_has_saturation` (S ≥ 80 が 20%+) を組み合わせ。
  `strict_pair_eyes=True` で水平ペア眼を必須化 (×印対策)。

テスト: `tests/test_patch_classifier.py` (15+ tests)。

#### `src/hybrid_classifier.py` — HybridClassifier

HSV + CNN を組み合わせる pipeline classifier。

- UI mask hit → EMPTY 強制
- CNN 確信度 ≥ `cnn_override_prob` (default 0.75、recognition_pipeline は **0.90**) → CNN
- 不確信 + HSV 一致 → CNN
- 不確信 + HSV 不一致 → **HSV 採用** (HSV の決定論性を優先)

`predict_proba_batch` で GPU 一括推論対応。

### 2.3 State Machine

#### `src/board_state_machine.py` — BoardStateMachine / BoardState / StateContext / DetectorSignals

1 サイドの状態遷移管理。`BoardState ∈ {MENU, STABLE, TSUMO_FALL, CHAIN, OJAMA_FALL, EFFECT}`。

- 試合外 (`is_match_active=False`) → 即 MENU、confirmed クリア。
- 検出器を順番評価、最初に発火した遷移先を採用。
- **NON-STABLE → STABLE 復帰**: `_merge_diff_only` で baseline と CNN の差分のみ反映
  (Phase C-5、CNN ぶれ取込防止)。
- `_apply_gravity_filter`: 浮きぷよ ban (空 cell 上の puyo は EMPTY に)。
- 連続 N frame (`stable_frame_count`、pipeline default 6) で同一 CNN 盤面 →
  pending → pending_count 達成で初回 STABLE 確定 (Phase C-7 E-1: 2 回目以降の
  CNN 多数決経路は停止、state 遷移時の物理推論のみで confirmed 更新)。
- `next_queue`: ネクストペア履歴 (最新 8 件)。

テスト: `tests/test_board_state_machine.py` (4+ tests)。

#### `src/state_detectors.py` — Chain/Effect/Ojama/Tsumo PhaseDetector

`StateTransitionDetector` Protocol を満たす 4 detector。

- **ChainPhaseDetector**: `signals.chain_event != None` → CHAIN; `event=None` &
  state==CHAIN → STABLE 復帰。
- **TsumoPhaseDetector**: 直近 STABLE との puyo 数差 +1〜+2 が連続 `consec_threshold=2`
  frame → TSUMO_FALL。**着地検出**: TSUMO_FALL 中に CNN 盤面が連続 `landed_consec=2`
  frame 同一 → STABLE 復帰。
- **OjamaPhaseDetector**: `score_delta` (相手 score 増分) ≥ 70 (= おじゃま 1 個分) →
  OJAMA_FALL。state==OJAMA_FALL なら次 frame で STABLE 復帰 (ロックイン回避)。
- **EffectPhaseDetector**: 現状 skeleton (常に None)。全消しテンプレート未統合。

テスト: `tests/test_state_detectors.py` (8+ tests)。

### 2.4 推論盤面と乖離検出

#### `src/inference_board.py` — InferenceBoardGenerator

state ごとに「現 frame で表示すべき推論盤面」を生成。

- STABLE → `confirmed_board` 直返し
- CHAIN → 最初の 1 度だけ `ChainSimulator.simulate(confirmed)` を実行、
  `(time-trigger)/(end-trigger)` 進行率で `steps[idx].board_after` を返す
- TSUMO_FALL/OJAMA_FALL/EFFECT → confirmed を hold (MVP)
- MENU → None

#### `src/drift_detector.py` — DriftDetector

inferred vs cnn の cell 単位差分を監視 (UNKNOWN は除外)。
`mismatch_count >= cell_threshold(=6)` が `frame_threshold(=3)` 連続で `needs_resync=True`。

→ pipeline は `state_machine.reset(keep_match_state=True) + drift.reset() + gen.reset()` で再同期。

テスト: `tests/test_drift_detector.py` (12+ tests)。

### 2.5 単一フレーム整合性 / ノイズ除去

#### `src/board_rules.py` — apply_gravity / clear_floating_above_gap

- `apply_gravity`: 列ごとに非空セルを下詰め (UNKNOWN 位置固定)。
- `clear_floating_above_gap(min_gap=2)`: スタックから 2+ 行空白で隔たれた上方
  非空セルを EMPTY に。連鎖中の浮き / UI overlay / ネクスト落下中の擬似 puyo 除去用。
  ImageReader.read_board の最終段で適用。

テスト: `tests/test_board_rules.py`。

#### `src/temporal_smoother.py` — TemporalSmoother

直近 N frame (`DEFAULT_WINDOW_SIZE=15` ≈ 0.5s @ 30fps) の per-cell 最頻値で平滑化。
0-9 の color code を `bincount` で集計、同票時は最小値。

注意: **現在の RecognitionPipeline は内部に独自 smoothing (`_smooth_board`,
default temporal_smoothing=1=無効)** を持つ。`TemporalSmoother` 単体は旧
`BoardRecognitionPipeline` 経由で使われる。

テスト: `tests/test_temporal_smoother.py` (10 tests)。

#### `src/stateful_board_tracker.py` — StatefulBoardTracker

物理ルールで遷移を選別する盤面追跡器:

- EMPTY → 色: 受理 (落下)
- 色 → EMPTY: chain_event (4+ 同色消失) のみ受理
- 色 → 異色: chain_event のみ受理
- 色 → OJAMA: 常に reject

旧 pipeline の安全装置。`expected_new_colors` 引数で next/dnext+既存色のみ受理する厳格モード対応。

テスト: `tests/test_stateful_board_tracker.py` (19+ tests)。

#### `src/ui_mask.py` — UiMaskMatcher

`models/ui_templates/` 配下の png に対して NCC マッチング (閾値 0.75)。
×マーク等の UI overlay を puyo 誤検出から除外する。
HybridClassifier / GatedCnnClassifier / ImageReader の各層で is_ui チェック。

#### `src/physics_sanity.py` — PhysicsSanityChecker

単一フレーム盤面に対する **検出のみ** (修正はしない):

- AIRBORNE: 非空 cell の真下が空 → 浮遊違反
- UNRESOLVED_CHAIN: 4+ 連結が消えていない → 連鎖未開始 or 色誤認

レビュー / 学習フィードバック用。runtime には組み込まれていない。
テスト: `tests/test_physics_sanity.py`。

#### `src/cell_recovery_refiner.py` — CellRecoveryRefiner (Phase Z-2)

cell 単位の検出漏れ・色誤認補正:

1. EmRecovery: recognized=EM/?? かつ HSV 高彩度 → HSV 主要色
2. OjmRecovery: recognized=EM/?? かつ低彩度+中明度 → OJM
3. HsvVote (HSV_VOTE_S_MIN=80): puyo 同士で色不一致なら HSV 高彩度側を採用

旧 StatePipeline 用。新 RecognitionPipeline には組み込まれていない (役割縮小予定、`recognition_strategy_pivot` 参照)。

### 2.6 隠し段 / 確率盤面

#### `src/hidden_row_inferrer.py` — infer_hidden_row / HiddenInferenceResult

prev_board → cur_board の差分とネクスト履歴 (落下色) から row 0 (隠し段) の
確率分布を推論。

- n=2 (新規 2 セル、ペア整合): 隠し段は EMPTY 確定
- n=1: もう 1 セルが隠し段にある → 列候補 (観測列または隣接列) で確率分布
- n=0/3+: 推論しない

テスト: `tests/test_hidden_row_inferrer.py` (7+ tests)。

#### `src/probabilistic_board.py` — ProbabilisticBoard / ProbabilisticCell

13×6 の cell に確率分布 (dict[color, prob]) を保持。CERTAIN_THRESHOLD=0.95 以上で確定扱い。

- `from_board(board)`: 既存 Board → 全セル確定 1.0 (UNKNOWN は均等分布)
- `to_board(threshold)`: 確定セル以外は UNKNOWN 化
- `to_max_likelihood_board()`: 最尤色採用 (UNKNOWN 化なし)
- `sample_board(rng)`: Monte Carlo サンプリング
- `total_uncertainty`, `n_uncertain`, `entropy`: 不確実性メトリクス

テスト: `tests/test_probabilistic_board.py` (11+ tests)。

#### `src/hidden_row_tracker.py` — 古い時系列推論ヘルパ

連続フレームの差分から隠し段を推定する旧モジュール。新 pipeline では
`hidden_row_inferrer` (確率版) に役割移行済。

### 2.7 ネクスト・スコア・連鎖

#### `src/next_detector.py` — NextDetector / NextDetectionResult

画面中央上部の next/dnext puyo (1P/2P 両方) を CNN+HSV ハイブリッドで分類。
ROI は 1920×1080 ハードコード v10 確定 (`ROI_1P_NEXT_TOP` など)。
2P 用は中央 x=960 で水平ミラー。背景色 (1P=水色, 2P=ピンク) 別の HSV ルール。

#### `src/match_state.py` — MatchStateDetector

盤面 ROI 上部 (row 2-5, col 1-4) の HSV V 平均で試合中/メニュー判定。
**閾値 IN_MATCH_V_MAX = 170.0** (試合中 V 69-156, 非試合 V 178-231)。

#### `src/win_panel.py` — WinPanelDetector

画面中央下部の "数値★ WIN ★数値" パネルを NCC マッチングで検出 (閾値 0.70)。
試合セクション境界判定用。

#### `src/score_zero.py` — ScoreZeroDetector

両側 score 表示の "00000000" を NCC マッチング (閾値 0.85) で検出。
試合切替判定の高信頼シグナル。

#### `src/match_winner.py` — MatchWinnerDetector

WIN パネル数値を 16×16 大津二値で 256-bit signature 化 → Hamming 距離で
変化判定 (DIGIT_DIFF_HAMMING=10, ASYMMETRY_RATIO=2.5)。OCR 不要。
video_02 で 50/50 全成功実績あり。

#### `src/score_ocr.py` — ScoreOcr / ScoreTracker / ScoreDelta

8 桁 score 表示を `models/ui_templates/score_digits/digit_0..9.png` の NCC で
読み取り (NCC_MIN_CONFIDENCE=0.55, AVG_MIN=0.65)。`ScoreTracker` は時系列差分。

#### `src/chain_detector.py` — VideoChainTracker / ChainEvent

連続 frame の puyo 数急減 (`ERASURE_MIN_DROP=4`) を検出 → 直前盤面で
`ChainSimulator.simulate` → ChainEvent (chain_count, total_score, ojama_sent etc.)。

注意: drop 観測 1 frame しか event を返さないため、pipeline 側で
`CHAIN_HOLD_PER_STEP_SEC=0.3 × chain_count` 秒間保持する。

### 2.8 動画別キャリブレーション / 学習

#### `src/per_video_model_selector.py` — select_phase_b_model / select_model_for_video

動画 ID から最適 CNN model パスを返す hardcoded mapping。

- v17b ベスト: video_01/04/05/06/10/11/16/17/19
- それ以外: v16
- Phase B mapping: `PHASE_B_CNN_BEST_VIDEOS={1,2,7,9,10,11,12,13,16}` で
  cnn_phase_b_v1.pt 採用、それ以外は HSV のみ。

#### `src/online_hsv_calibrator.py` — OnlineHsvCalibrator

未知動画でのリアルタイム HSV 範囲学習 (`unknown_video_realtime_hsv` 段階 2)。

- HIGH_CONF=0.99 + HSV 単独判定一致 + 連続 stable 条件で信頼サンプル抽出
- 色別 EMA (alpha=0.05) で平均 / 分散更新
- N≥200 サンプルで動画別 ranges 採用 (mean ± 1.5σ)
- `ImageReader.set_color_ranges_from_simple` で classifier に注入

**ステータス: モジュール実装済みだが Pipeline には未統合**。
テスト: `tests/test_online_hsv_calibrator.py`。

#### `src/calibration.py` — CalibrationHelper / CalibratedConfig

annotation.json (盤面コーナー + 色サンプル位置) → BoardRegion + HsvRange を
自動生成し JSON 永続化。`MatchStateDetector.load_default` 等から読み込む。

### 2.9 動画 / ストレージ

#### `src/storage.py` — StorageManager / VideoRecord

動画 URL 履歴 (`data/video_history.json`) 管理。`cleanup()` で動画 mp4 削除
(URL とメタデータは永続)。

---

## 3. 全体パイプラインのフロー詳細

`RecognitionPipeline.update(frame_idx, t_sec, frame)` の擬似コード
(`src/recognition_pipeline.py:382-570`):

```
def update(frame_idx, time_sec, frame):

    # ----- Step 1: 試合状態判定 + hysteresis -----
    raw_active = MatchStateDetector.detect(frame).IN_MATCH or force_in_match
    if not raw_active and (score_tracker_1p.last>0 or score_tracker_2p.last>0):
        raw_active = True   # score-based 補強
    recent_active = (frame_idx - last_active_frame_idx) <= MATCH_ACTIVE_HOLD_FRAMES(=10)
    sm_active = (sm_1p.state in NON-MENU) or (sm_2p.state in NON-MENU)
    is_active = raw_active or recent_active or sm_active
    # 失敗時 fallback: is_active=False (=MENU) は state machine が confirmed クリア

    # ----- Step 2: CNN raw 盤面取得 -----
    cnn_1p_raw, cnn_2p_raw = ImageReader.read_both_boards(frame)
       # 内部: 1080p resize → MatchStateDetector (use_match_state=False で skip)
       #       → for each region (P1, P2):
       #            - HSV cvtColor (BG FP 用に共有)
       #            - for each cell (78 個):
       #                - cell_sample_rect で patch 切出 (中央 50%)
       #                - BG fingerprint で "空 cell" 早期判定
       #                - 残った patch を classify_batch (HybridClassifier 5-20x)
       #            - clear_floating_above_gap(min_gap=2)  ← 浮き ban
       #            - _infer_hidden_rows  ← 重力推論で row 0 を確定/UNKNOWN
       # 失敗時 fallback: 例外時は呼び出し元で frame skip 想定 (現状無し)

    # ----- Step 2b: 背景 FP 自動採取 -----
    if is_active and (frame_idx - match_active_started) >= 5 and not bg_fp_captured:
        if cnn_1p_raw.count_puyos() + cnn_2p_raw.count_puyos() == 0:
            bg_frame_buffer.append(frame.copy())
            if len(buffer) >= 5:
                bg_fp_p1, bg_fp_p2 = capture_robust_fingerprint(median frames)
                ImageReader.set_background_fingerprints(bg_fp_p1, bg_fp_p2)

    # ----- Step 3: temporal smoothing (default no-op) -----
    cnn_history_1p.append(cnn_1p_raw)
    if smoothing_n > 1: cnn_1p = _smooth_board(history)  # majority vote
    else: cnn_1p = cnn_1p_raw

    # ----- Step 4: 連鎖検出 + chain_event hold -----
    chain_banned = (frame_idx - match_active_started) < CHAIN_BAN_FRAMES_AFTER_MATCH_START(=30)
    ev = chain_tracker_1p.update(time_sec, cnn_1p)
    if ev and not chain_banned:
        active_chain_1p = ev
        chain_until_1p = time_sec + 0.3 * ev.chain_count
    chain_ev_1p = active_chain_1p if time_sec < chain_until_1p else None

    # ----- Step 5: score 差分 + next pair -----
    score_d_1p = score_tracker_1p.update(frame).delta
    score_d_2p = score_tracker_2p.update(frame).delta
    next_pair_1p, next_pair_2p = next_detector.detect_both(frame)

    # ----- Step 6: side ごとに state machine + 推論 + drift -----
    for side, sm, gen, drift in [(1P, sm_1p, gen_1p, drift_1p), (2P, ...)]:

        signals = DetectorSignals(
            time_sec, cnn_board, is_match_active=is_active,
            chain_event=chain_ev_{side},
            score_delta=score_d_{相手 side},   # OJAMA は相手 score 増分で発火
            next_pair=next_pair_{side},
        )

        prev_state = sm.context.state
        prev_confirmed = sm.context.confirmed_board.copy()
        prev_next_queue = list(sm.context.next_queue)

        ctx = sm.update(frame_idx, signals)
            # 試合外 → MENU (confirmed クリア)
            # 各 detector 順 (Chain → Effect → Ojama → Tsumo) で遷移先返した最初を採用
            # NON-STABLE → STABLE 復帰時は _merge_diff_only で baseline 差分のみ反映

        # TSUMO_FALL → STABLE 復帰時 (Option C-2/E-2)
        if prev_state == TSUMO_FALL and ctx.state == STABLE:
            falling_pair = prev_next_queue[-2]  # TSUMO 開始時の next = 落下中
            inferred_landing = _compute_landing_inferred(prev_confirmed, cnn_board, falling_pair)
            # 位置: CNN 差分 (空→puyo の cell 抽出、最も低い 2 cell が着地点)
            # 色: 縦置きなら next_pair[top]→next_pair[bot]、横置きは CNN
            if falling_pair でぷよ色なら:
                pboard = infer_hidden_row(prev_confirmed, inferred_landing, falling_pair)
                # 隠し段 row 0 を確率分布で更新、p>=0.95 なら確定、それ未満は UNKNOWN
            ctx.confirmed_board = inferred_landing
            _inject_pseudo_chain_event(side, time_sec, inferred_landing)
                # ChainSimulator.simulate(landing) で chain_count >= 1 なら疑似 ChainEvent

        # CHAIN → STABLE 復帰時 (Phase C-6 C)
        if prev_state == CHAIN and ctx.state == STABLE and chain_event:
            cr = ChainSimulator.simulate(chain_event.before_board)
            ctx.confirmed_board = cr.final_board (gravity 適用)

        inferred = InferenceBoardGenerator.generate(ctx, chain_event, time_sec)
            # STABLE: confirmed_board そのまま
            # CHAIN: simulate progress マッピングで steps[idx].board_after
            # NON-STABLE: confirmed を hold (MVP)
            # MENU: None

        drift_res = DriftDetector.update(inferred, cnn_board)
            # mismatch >= 6 cell が 3 frame 連続で needs_resync
        if drift_res.needs_resync:
            sm.reset(keep_match_state=True)
            drift.reset()
            gen.reset()

        # W-α: STABLE で prob_board を publish
        publish_prob_board = side_prob_board or ProbabilisticBoard.from_board(confirmed)

        return SideResult(side, state, cnn_board, inferred, confirmed,
                          drift_res, score, score_delta, chain_event, prob_board)

    return PipelineResult(frame_idx, time_sec, is_active, p1, p2)
```

### Fallback まとめ

| 失敗 | Fallback |
|---|---|
| MatchStateDetector が UNKNOWN | hysteresis (last_active_frame_idx) で吸収 |
| score OCR 読めない (NCC < 0.55) | delta=0 (OJAMA 発火しない) |
| next_detector load fail | next_pair=None (隠し段推論スキップ) |
| BG FP 採取失敗 | bg_fingerprint=None (HSV/CNN だけで判定) |
| ChainSimulator 失敗 | confirmed 上書きスキップ (CNN 経由で merge_diff_only) |
| infer_hidden_row 例外 | side_prob_board=None → from_board fallback |
| drift needs_resync | sm.reset(keep_match=True) + drift.reset() で再同期 |

---

## 4. State Machine (BoardState) 遷移詳細

### 4.1 各 state の意味と confirmed 更新有無

| State | 意味 | confirmed 更新? | 採用ロジック |
|---|---|---|---|
| MENU | 試合外 (タイトル / リザルト / リトライ) | × クリア | match_state IN_MATCH 失敗時 |
| STABLE | 平常時 | ○ (初回のみ多数決, 以降は state 遷移時の物理推論) | CNN 連続 N=6 frame 一致 |
| TSUMO_FALL | ツモ落下中 | × | puyo 数 +1〜+2 が連続 2 frame |
| CHAIN | 連鎖中 (消去 + 重力) | × | VideoChainTracker の event |
| OJAMA_FALL | おじゃま落下中 | × | 相手 score_delta ≥ 70 |
| EFFECT | 全消し演出 / 連鎖カットイン等 | × | 現状未実装 (skeleton) |

### 4.2 遷移条件 (具体的に何を見るか)

```
[MENU] ──── pending_count >= 6 (連続 6 frame 同一 CNN 盤面) ────► [STABLE 初回]

[STABLE] ─┬── chain_event != None ──────────────────────────────► [CHAIN]
          ├── score_delta >= 70 ────────────────────────────────► [OJAMA_FALL]
          ├── puyo 数 +1〜+2 が連続 2 frame ───────────────────────► [TSUMO_FALL]
          └── 試合 active 終了 ───────────────────────────────────► [MENU]

[TSUMO_FALL] ─┬── CNN 盤面が連続 2 frame 同一 (着地確定) ─────────► [STABLE]
              │   └── pipeline: _compute_landing_inferred で
              │       位置=CNN 差分, 色=ネクスト履歴で confirmed 上書き
              │       infer_hidden_row で隠し段確率推論
              │       _inject_pseudo_chain_event (連鎖発火物理推論)
              ├── puyo 数 0 (連鎖発火による消去) ────────────────► [STABLE]
              └── chain_event 観測 ──────────────────────────────► [CHAIN]

[CHAIN] ─── chain_event=None (CHAIN_HOLD 過ぎ) ────────────────► [STABLE]
            └── pipeline: ChainSimulator.simulate.final_board で confirmed 上書き

[OJAMA_FALL] ── 次 frame で必ず ─────────────────────────────► [STABLE]
                (ロックイン回避、本格落下完了検出は B-3/B-4 統合で)

[EFFECT] ── (現状遷移なし、skeleton) ────────────────────────────────────
```

### 4.3 主要パラメータ

| パラメータ | 値 | 場所 | 意味 |
|---|---|---|---|
| `stable_frame_count` | 6 | `RecognitionPipeline.load_default` | STABLE 確定の連続 frame 数 |
| `temporal_smoothing` | 1 | 同上 | CNN 多数決 window (1=無効) |
| `consec_threshold` | 2 | `TsumoPhaseDetector` | TSUMO 進入の連続 frame 数 (CNN ぶれ吸収) |
| `landed_consec` | 2 | `TsumoPhaseDetector` | TSUMO 着地確定の連続 frame 数 |
| `score_threshold` | 70 | `OjamaPhaseDetector` | OJAMA 発火 score 増分 (= 70 点 = おじゃま 1 個) |
| `CHAIN_HOLD_PER_STEP_SEC` | 0.3 | `RecognitionPipeline` | CHAIN state ロック秒/連鎖段 |
| `MATCH_ACTIVE_HOLD_FRAMES` | 10 | 同上 | 試合 active hysteresis frame |
| `CHAIN_BAN_FRAMES_AFTER_MATCH_START` | 30 | 同上 | 試合開始直後 CHAIN ban frame |
| `cnn_override_prob` | 0.90 | HybridClassifier | CNN > HSV の確信度閾値 |
| `DRIFT_CELL_THRESHOLD` | 6 | DriftDetector | 1 frame の不一致 cell 数 |
| `DRIFT_FRAME_THRESHOLD` | 3 | DriftDetector | 連続 drift frame 数で resync |
| `CERTAIN_THRESHOLD` | 0.95 | ProbabilisticBoard | 確率盤面の確定閾値 |
| `IN_MATCH_V_MAX` | 170.0 | MatchStateDetector | 試合中判定の HSV V 上限 |

---

## 5. 時系列ノイズ除去メカニズム

「2 frame 連続誤認」「序盤数手の誤認識」への耐性を分析する。

### 5.1 既存メカニズム

| レイヤー | モジュール | 効果 | 副作用/限界 |
|---|---|---|---|
| (a) Cell-level バッチ | HybridClassifier (override 0.90) + GatedCnnClassifier の HSV strong color override (赤・紫) + UI mask | 単 frame の CNN 1 cell 誤判定をルールで上書き | `cnn_override_prob=0.90` 未満では HSV 採用 → HSV ルール外の動画別色で抜けやすい |
| (b) Single-frame physics | `clear_floating_above_gap(min_gap=2)` (image_reader 内) | 浮きぷよ ban (UI overlay, transient state, 落下中 puyo) | 連鎖中の物理的浮きを誤って消す可能性 |
| (c) 単一 frame integrity | `_infer_hidden_rows` (空列 → 隠し段空) | 13 段目の物理推論 | 可視最上段に puyo があると UNKNOWN 化 |
| (d) Temporal smoothing | `RecognitionPipeline._smooth_board` (default N=1, 無効) / 旧 `TemporalSmoother` | per-cell N frame 多数決 | **default 無効**。N=15 で 0.5s 遅延 → state machine 反応速度低下 |
| (e) State machine 連続多数決 | `BoardStateMachine._update_within_current_state` | 連続 N=6 frame 同一で初回 STABLE 確定 | 初回のみ。**2 回目以降の confirmed 更新には使われない**(Phase C-7 E-1) |
| (f) State 遷移時の物理推論 | `_compute_landing_inferred` (TSUMO→STABLE), ChainSimulator (CHAIN→STABLE) | 着地時の CNN cell 値を排除し、**位置=差分・色=ネクスト履歴**で確定 | ネクスト履歴が無いと CNN 値 fallback (色誤認 残存) |
| (g) `_merge_diff_only` | NON-STABLE→STABLE 復帰時 baseline 維持 | CNN ぶれが confirmed に乗るのを構造的に防ぐ | baseline が誤ると永続化 |
| (h) `_apply_gravity_filter` | 浮き ban (state machine 内) | (b) と同等を transition 後にも適用 | (b) 同様 |
| (i) Drift detector | `DriftDetector` | 推論 vs CNN の連続 3 frame ≥6 cell 乖離で resync | 6 cell 閾値を下回る微小 drift は検出不能 |
| (j) Stateful tracker | `StatefulBoardTracker` (旧 pipeline) | 物理ルール違反 (色→OJAMA, 色→異色 chain なし) reject | **新 RecognitionPipeline には組込まれていない** |

### 5.2 「置いた直後の 2 frame 誤認」が漏れる経路

ユーザー指摘の「置いた後の一瞬誤認」が **既存メカニズムを突破する** ケース:

1. TSUMO_FALL → STABLE 着地時に `_compute_landing_inferred` が動くが,
   **着地後 1〜2 frame のあと、新しい STABLE で CNN が一時的に違う色を返した場合**:
   - state は STABLE のまま、Phase C-7 E-1 仕様で「2 回目以降の CNN 多数決経路で
     confirmed 更新は停止」しているので、confirmed は **理論的には保護される**。
   - **しかし** drift detector が連続 3 frame ≥6 cell 違うと判定すると `needs_resync`
     が発火し、reset で confirmed=None に戻り、次の連続 6 frame 多数決で誤色が確定する。
2. 「2 回目以降の confirmed 更新は state 遷移時のみ」だが,
   TSUMO の `consec_threshold=2` は 2 frame 連続で puyo +1〜+2 を要求。
   置いた直後 1 frame が雑音 (例: 着地アニメで CNN が空 cell に色を読む) でも,
   **雑音が 2 frame 連続なら TSUMO_FALL 誤判定 → 着地推論で誤色確定** がありえる。
3. CNN そのものの 99.45% 精度 → 平均 78 cell × 0.55% ≈ 0.43 cell/frame 誤り。
   2 frame 連続で同じ cell が同じ誤色を返す確率 ≈ 0.0024% × cell数 が漏れの主源。

### 5.3 「序盤数手の認識ミス」が起きる経路

1. 試合開始 +5 frame で BG fingerprint 採取 → puyo 0 個 5 frame 蓄積後に確定。
   この間 **BG FP 無し** で HSV/CNN だけで判定 → 動画別 BG 色 ≠ default で空 cell が
   誤色になりやすい。
2. CHAIN_BAN 30 frame は連鎖誤検出 ban のみ。**TSUMO 誤検出 ban は無い** ので、
   試合開始直後の puyo 出現 (= 0 → 2 個) を **TSUMO_FALL と誤認** することがあり,
   _compute_landing_inferred が走るが prev_confirmed=None または初期空盤面で
   spurious 着地推論される。
3. 試合開始時のネクスト履歴 (`next_queue`) は最初の 1〜2 試行はまだ蓄積されていない。
   `falling_pair=None` → CNN 値 fallback で色誤認が残る。
4. Online HSV calibrator (`OnlineHsvCalibrator`) は **モジュール実装済みだが
   Pipeline には未統合**。training に無い動画では default HSV で全試合進行。

### 5.4 「何を防げて、何を防げないか」

| 状況 | 防げる | 防げない |
|---|---|---|
| 1 frame の独立 CNN ぶれ | (a) override + (e) 連続多数決 + (i) drift | 1 frame の過半数 cell 同時誤りなら drift 発火、resync 後 confirmed が誤る |
| 2 frame 連続同一誤認 | (i) drift_consec=3 でまだ resync しない | (e) は初回のみ、その後の confirmed 更新は state 遷移時物理推論依存 |
| 連鎖中の残光・エフェクト | (f) ChainSimulator 上書き | ChainSimulator.simulate が誤った場合 (rare) |
| UI overlay (×印) | (a) UiMaskMatcher (NCC 0.75) | テンプレートに無い UI、低マッチング (映像差別の動画) |
| 試合開始直後のメニュー誤判定 | hysteresis (10 frame), score>0 補強, sm_active 強制 | 全 detector が同時に誤判定する場合 |
| 動画別色ぶれ | per_video_model_selector (1〜19) | 未知動画 (v20+) は default v16/HSV で進行 |
| 隠し段 (row 0) 認識 | (c) `_infer_hidden_rows` + `infer_hidden_row` 確率推論 | ネクスト履歴不足、回し入れの異常パターン |
| 連鎖 VideoChainTracker 漏れ | `_inject_pseudo_chain_event` 二重保険 | ChainSimulator が連鎖 0 と判定したケース |

---

## 6. 学習で達成した認識精度

(`memory/project_phase_h1_results.md`, `memory/feedback_recognition_target_995.md`,
`docs/HANDOFF_2026-05-03_PHASE_B.md` より)

### 6.1 過去の達成値

| 段階 | 構成 | Holdout / 認識率 | 備考 |
|---|---|---|---|
| Phase U v6 | cnn_phase_u_v6.pt | 99.45% (cell holdout) | 2026-04-29 |
| Phase B v1 | cnn_phase_b_v1.pt = phase_u + menu_truth | **99.61%** (cell holdout) | 2026-05-03、現在最良 |
| Phase B drift v2 (失敗) | + drift_truth | 63.64〜93.31% | drift_truth は CNN 訓練に使えないと確定 (B-13) |

### 6.2 動画別精度 (Phase B)

`data/phase_b_eval_summary*.tsv` より:

| 構成 | 平均 STABLE % | drift | vs HSV |
|---|---|---|---|
| HSV のみ | 53.5 | 25.2 | — |
| CNN-v1 一律 | 52.7 | 26.3 | -0.8pt |
| Per-Video (PV) | 58.3 | 27.0 | +4.8pt |
| Smoothing=3 一律 | 56.6 | 26.9 | +3.1pt |
| **PV2 (PV + per-video smoothing)** | **60.4** | 27.5 | **+6.9pt** |

**単 frame 精度 (drift 解析、試合中区間)**:
- 平均 ~96.7%、最低 v15=95.15%、最高 v11=98.84%。
- 試合前 (v14, v17, v18): 100%。
- cell 位置: r1-r6 c2-c4 (盤面中央上部) で drift 集中、r12 (最下段) は最安定。
- color confusion: **EM ↔ X (任意色)** が支配的 (top: EM→BL 880件, BL→EM 509件)。

### 6.3 動画別 / モデル使い分け (`per_video_model_selector.py`)

```
V17B_BEST_VIDEOS    = {1, 4, 5, 6, 10, 11, 16, 17, 19}    → cnn_phase_u_v17b.pt
PHASE_B_CNN_BEST_VIDEOS = {1, 2, 7, 9, 10, 11, 12, 13, 16}   → cnn_phase_b_v1.pt
PHASE_B_HSV_BEST_VIDEOS = {3, 4, 5, 6, 8, 15, 19}            → HSV のみ
PHASE_B_SMOOTHING3_BEST_VIDEOS = {2, 3, 7, 8, 10, 11, 15, 16}
```

未知動画 (v20+) は default v16 + HSV で進行。

### 6.4 99.99% への ギャップ

`feedback_recognition_target_995.md` より:
- **目標**: STABLE state 確定盤面で 99.99% (単 frame ではなく)。
- 連続 N=6 frame 多数決 → 独立 frame 誤りは確率的に消える計算 (理論上 99.45% × 6 frame で 99.99%+)。
- 実測 PV2 STABLE 60.4% は「STABLE 確定できた frame の比率」であり、その中の精度は別軸。
- **実装ギャップ**: 現状 confirmed 更新は state 遷移時物理推論依存 (Phase C-7 E-1)
  になっているため、初回 STABLE 確定後は CNN 多数決の恩恵を受けず、
  TSUMO/CHAIN 推論が誤ると confirmed に永続化する。

---

## 7. 既知の認識弱点

ユーザー指摘 + memory + コード読解から抽出。

### 7.1 序盤 (試合開始 +20s 以内) の puyo 認識

- **症状**: cell が連続 EMPTY のまま、または初手が誤色で確定。
- **原因**:
  - BG fingerprint 採取が試合開始 +5 frame 以降 (puyo 0 個 5 枚必要) → 採取前は default HSV のみ。
  - ネクスト履歴 (`next_queue`) が未蓄積 → `_compute_landing_inferred` で CNN 色 fallback。
  - 初回 STABLE 確定は連続 6 frame 同一が条件 → 動画別ぶれで pending_count リセットが頻発。
  - Online HSV calibrator は **未統合** (段階 2)。
- **関連 memory**: `project_unknown_video_realtime_hsv.md`。

### 7.2 置いた直後の数 frame 誤認 (2 frame 連続誤認)

- **症状**: 置いた瞬間〜着地直後に CNN が違う色を返し、それが confirmed に永続化。
- **原因**:
  - TsumoPhaseDetector の `consec_threshold=2` は 2 frame 連続誤認に対して防御不能。
  - `_merge_diff_only` は puyo→空遷移を ban するが,
    **空→誤色は ban していない** (allow_puyo_to_empty=True default 経路)。
  - DriftDetector は 6 cell 以上の乖離 + 3 frame 連続で発火 → 数 cell の誤りは検出不能。
  - **Phase C-7 E-1**: 2 回目以降の CNN 連続多数決は confirmed 更新に使われない →
    state 遷移時の物理推論で 1 度誤ると、次の state 遷移まで誤色のまま固定。

### 7.3 おじゃまぷよと puyo の境界

- HSV 範囲: OJAMA は `S < OJAMA_S_THRESHOLD(60) and V > OJAMA_V_MIN(100)`
  (`image_reader.py:44-48`)。低彩度の黄ぷよが OJAMA に流れ込みやすい。
- 緩和策: `CellRecoveryRefiner.OJM_S_MAX=50, V_MIN=145, V_MAX=210` (Z-2)。
- 新 pipeline には **CellRecoveryRefiner 未統合**。

### 7.4 隠し段 (row 0) の量子状態

- `_infer_hidden_rows`: 可視最上段が空なら隠し段も EMPTY 確定、それ以外は UNKNOWN。
- `infer_hidden_row` (新): TSUMO→STABLE 復帰時にネクスト履歴ベースで確率分布。
- `prob_board` は STABLE で publish されるが,
  **下流 indicator 計算 (`compute_all_probabilistic`) で確率版が使われているか要確認**
  (コード未確認、`recognition_pipeline.py:42-50` のコメントは「下流 phase_e_collect 側で扱う」と記載)。

### 7.5 エフェクト (連鎖光、全消し演出)

- `EffectPhaseDetector` は **skeleton (常に None)** → 全消しエフェクトを EFFECT state に
  遷移できない。
- 代替: VideoChainTracker の chain_event が CHAIN state にロックして連鎖アニメ期間
  をカバー (`CHAIN_HOLD_PER_STEP_SEC * chain_count`)。
- 全消しは ChainEvent.is_all_clear で判定可、ただし state 遷移には未使用。

### 7.6 背景透過 / UI 重なり (×マーク等)

- UiMaskMatcher (`models/ui_templates/`, NCC 0.75) で対応。
- HybridClassifier / GatedCnnClassifier / ImageReader の各層で is_ui チェック。
- 制限: テンプレートに無い UI (テロップ、新コラボ装飾) は対応不可。
- TelopDetector はオプションで実装済 (`use_telop_mask=True` で有効化、default OFF)。

### 7.7 動画別 HSV 範囲ズレ

- `per_video_model_selector` は **1〜19 のみ hardcoded mapping**。
- 未知動画は default v16 / HSV → 色再現が異なる動画 (新解像度・新放送) で精度低下。
- `OnlineHsvCalibrator` 実装済だが pipeline 未統合 → 改善ポイント。

### 7.8 その他

- **動画リサイズの劣化**: `read_both_boards` が 1080p に強制リサイズ → 720p/4K 動画で
  cv2.INTER_AREA 経由の色情報損失。
- **MatchStateDetector の混雑誤判定**: `IN_MATCH_V_MAX=170.0` で
  混雑時 V=151-156 の試合中盤面はギリギリ通る。マージン 14 ポイント。
- **scoreOcr の連鎖中誤読**: 連鎖中の "+1240" 等で 8 桁読めると score 急増 → OJAMA 誤発火可能。
  `NCC_AVG_MIN_CONFIDENCE=0.65` で対策済。

---

## 8. Phase I 改善ロードマップ (提案)

ユーザー指摘 (一瞬誤認 + 序盤未認識) を踏まえた改善案 5 件。

### 案 1: Online HSV calibrator の Pipeline 統合 (★ 本命 ★)

**何をする**:
- `OnlineHsvCalibrator` を `RecognitionPipeline` に統合。
- 試合中の信頼サンプル (CNN 確信度 ≥ 0.99 + HSV 一致 + 連続 stable) で動画別 HSV 範囲を学習し,
  `ImageReader.set_color_ranges_from_simple` で classifier に注入。
- 既知動画では training set 由来 HSV で初期値を与え、未知動画は warmup 1〜2 試合で動画別最適化。

**工数**: 中 (2-3 日)。モジュールは存在するため統合 + テストのみ。
**期待効果**: 未知動画の認識率を default HSV → 動画別 HSV に近づける。
training set で 99.4% 平均だった精度を未知動画でも保持。
**リスク**: 信頼サンプル抽出条件が緩いと汚染で誤った HSV 範囲を学習しうる。
mitigation: Phase Z-3I 改の HIGH_CONF=0.99 / MIN_SAMPLES=200 で十分厳格。

### 案 2: 試合開始 BG キャリブの強化 (序盤対策)

**何をする**:
- 現状: BG FP 採取は試合開始 +5 frame 以降、puyo 0 個 5 枚で `capture_robust_fingerprint`。
- 改善案 a: 試合開始 **前** (メニュー画面終了直後) からキャリブ用 frame を蓄積開始。
- 改善案 b: 連続 puyo 0 frame の代わりに「puyo 数 ≤ 2」でも採取 (2 個程度のツモなら BG cell は
  74〜76 個分残るので robust median が有効)。
- 改善案 c: 採取後も EMA で BG FP を継続更新 (動画 brightness 変化追従)。

**工数**: 小 (1 日)。`_bg_frame_buffer` ロジック修正のみ。
**期待効果**: 試合開始 +1〜+5 frame の BG FP 不在期間を解消、序盤誤認低減。
**リスク**: メニュー frame の取込で BG 学習が汚染。 mitigation: match_state IN_MATCH 確定後のみ採取。

### 案 3: 序盤認識の堅牢化 (state machine パラメータ強化)

**何をする**:
- `stable_frame_count` を 6 → 12 (試合開始 +60 frame は厳格化、それ以降は 6 に戻す)。
- `consec_threshold (TSUMO)` を 2 → 3 (試合開始 +60 frame の TSUMO 誤検出抑制)。
- 試合開始直後限定で **「CNN+HSV 一致時のみ採用」** のゲート追加 (cnn_override_prob を一時的に 1.0 に)。
- TSUMO_FALL 進入時に `prev_confirmed=None` または puyo 数 < 4 なら _compute_landing_inferred を
  spurious として **拒否** (= TSUMO_FALL 進入をブロック)。

**工数**: 小 (1 日)。RecognitionPipeline + state_detectors の閾値 + condition 追加。
**期待効果**: 序盤の TSUMO 誤判定 / 着地推論誤差を ban、序盤 confirmed の精度向上。
**リスク**: 厳格化で本来の TSUMO を見逃す。 mitigation: 試合開始 +60 frame 限定の時限式に。

### 案 4: prob_board の効果検証と下流活用

**何をする**:
- 現状: `RecognitionPipeline` は STABLE で `prob_board` を publish するが,
  下流 (indicator 計算、scorer) でどこまで活用されているか **要確認**。
- `compute_all_probabilistic` (場所要確認) で prob_board が使われているか測定し,
  使われていない指標があれば確率版指標を追加。
- evaluation: prob_board あり vs なしで PhaseAware overall を比較。

**工数**: 中 (2 日)。コード調査 + ベンチマーク。
**期待効果**: 隠し段の確率を活用することで、回し入れ多発動画 (上級者対戦) で
indicator 精度向上。直接的な認識率改善ではないが、認識不確実性を下流で活かせる。
**リスク**: 既存 indicator との互換性破壊。 mitigation: 新 indicator 追加で切替。

### 案 5: 「一瞬誤認」検出ヒューリスティック

**何をする**:
- 物理推論で「**前 frame の puyo が瞬間消えて違う色で復活**」を検知し reject。
- 具体: 直近 N frame (N=5) の per-cell 履歴で
  `[red, red, blue, red, red]` のような単発スパイクを検出 → 中央 frame の blue を red に修正。
- `StatefulBoardTracker` の物理ルール (色→異色 reject unless chain_event) を新 pipeline に
  再統合する案。

**工数**: 中 (2-3 日)。新規 SpikeFilter モジュール + RecognitionPipeline 統合。
**期待効果**: 2 frame 単発誤認を構造的に除去 (ユーザー指摘の中核問題への直接対策)。
**リスク**: 真の連鎖発火直前 frame が「単発スパイク」と誤認される可能性。
mitigation: chain_event=None 中のみ filter 適用、CHAIN state 中はスキップ。

### 改善案サマリ

| # | 案 | 工数 | 期待 | 主リスク |
|---|---|---|---|---|
| 1 | Online HSV calibrator 統合 | 中 (2-3d) | ★★★★ | 汚染学習 |
| 2 | 試合開始 BG キャリブ強化 | 小 (1d) | ★★★ | メニュー汚染 |
| 3 | 序盤認識パラメータ強化 | 小 (1d) | ★★★ | 真の TSUMO 見逃し |
| 4 | prob_board 効果検証 | 中 (2d) | ★★ (測定要) | 互換性破壊 |
| 5 | 一瞬誤認検出 | 中 (2-3d) | ★★★★ | 真連鎖直前誤検 |

**推奨優先順**: 5 → 3 → 1 → 2 → 4。
理由: 5 が「2 frame 連続誤認」への直接対策で最大効果、3 は 1 日で序盤強化、
1 は本命だが統合工数あり、2 は 1 と組合せ、4 は測定先行。

---

## 9. 参考資料 / 関連コード

### 9.1 主要ソース (絶対パス)

| 役割 | パス |
|---|---|
| 統合 pipeline | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\recognition_pipeline.py` |
| 旧 pipeline | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\board_recognition_pipeline.py` |
| ROI / HSV 分類 | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\image_reader.py` |
| Patch 分類器 | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\patch_classifier.py` |
| HSV+CNN ハイブリッド | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\hybrid_classifier.py` |
| State machine | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\board_state_machine.py` |
| State detectors | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\state_detectors.py` |
| 推論盤面 | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\inference_board.py` |
| Drift detector | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\drift_detector.py` |
| Temporal smoothing | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\temporal_smoother.py` |
| Stateful tracker | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\stateful_board_tracker.py` |
| 物理ルール | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\board_rules.py` |
| Physics sanity | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\physics_sanity.py` |
| UI mask | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\ui_mask.py` |
| Cell refiner | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\cell_recovery_refiner.py` |
| 隠し段確率 | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\hidden_row_inferrer.py` |
| 確率盤面 | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\probabilistic_board.py` |
| Next 検出 | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\next_detector.py` |
| Match state | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\match_state.py` |
| Win panel | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\win_panel.py` |
| Score zero | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\score_zero.py` |
| Match winner | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\match_winner.py` |
| Score OCR | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\score_ocr.py` |
| Chain detector | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\chain_detector.py` |
| Per-video selector | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\per_video_model_selector.py` |
| Online HSV (未統合) | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\online_hsv_calibrator.py` |
| Calibration | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\calibration.py` |
| Storage | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\src\storage.py` |

### 9.2 関連 memory (`~/.claude/projects/.../memory/`)

| memory file | 内容 |
|---|---|
| `feedback_recognition_target_995.md` | 99.99% 認識目標、state 確定盤面で評価 |
| `project_recognition_strategy_pivot.md` | state machine 主軸 pivot (2026-05-03) |
| `feedback_chain_phase_physics_only.md` | アクション中は全て物理推論 (2026-05-03 拡張) |
| `project_unknown_video_realtime_hsv.md` | Online HSV 三段階設計 |
| `project_phase_h1_results.md` | Phase H1 結果 (2026-05-08) |
| `project_handoff_2026-05-07_bc.md` | BCD 完了引継ぎ |
| `project_phase_e_completion.md` | Phase E 完成 (overall 0.659) |
| `feedback_priority_overlay_vs_rl.md` | RL 優先・オーバーレイ後回し |
| `reference_puyo_ai_recognition.md` | 先行研究: 既存 Puyo AI 認識手法 |

### 9.3 関連 docs

| docs file | 内容 |
|---|---|
| `docs/HANDOFF_2026-05-03_PHASE_B.md` | Phase B 完了引継ぎ (state machine 導入) |
| `docs/HANDOFF_2026-05-02.md` | Phase Z 完了 + ロジックレビュー |
| `docs/CURRENT_RECOGNITION_LOGIC.md` | (古い) 認識ロジック概観 |
| `docs/DETECTION_OVERVIEW_2026-04-30.md` | (古い) 検出概観 |
| `docs/PHASE_Z_FINAL_REPORT_2026-05-01.md` | Phase Z 最終レポート |
| `docs/SESSION_HANDOFF_2026-05-06.md` | 直近引継ぎ |

### 9.4 主要テスト数

`tests/` 配下計 121 ファイル / 733 テスト + 1 skipped 全パス。本資料関連:

| テストファイル | テスト数 |
|---|---|
| `test_recognition_pipeline.py` | 6 |
| `test_image_reader.py` | 4 |
| `test_image_reader_golden.py` | (未測定) |
| `test_patch_classifier.py` | 15 |
| `test_board_state_machine.py` | 10 |
| `test_state_detectors.py` | 8 |
| `test_temporal_smoother.py` | 5 |
| `test_stateful_board_tracker.py` | 19 |
| `test_drift_detector.py` | 12 |
| `test_hidden_row_inferrer.py` | 7 |
| `test_probabilistic_board.py` | 11 |
| `test_inference_board.py` | (未測定) |
| `test_online_hsv_calibrator.py` | (未測定) |

---

## 付録: 重要な定数早見表

```python
# image_reader.py
CELL_SAMPLE_RATIO = 0.5             # cell 中央 50% をサンプル
EMPTY_V_THRESHOLD = 40              # V < 40 で空
OJAMA_S_THRESHOLD = 60              # S < 60 でおじゃま候補
OJAMA_V_MIN = 100                   # V > 100 でおじゃま
RED_GREEN_DIFF_FOR_RED = 80         # 赤 vs 黄 の BGR R-G 差
DEFAULT_P1_REGION = (282, 160, 384, 720)
DEFAULT_P2_REGION = (1258, 160, 384, 720)

# board_state_machine.py
DEFAULT_STABLE_FRAME_COUNT = 6      # STABLE 確定の連続 frame 数
NON_STABLE_STATES = {TSUMO_FALL, CHAIN, OJAMA_FALL, EFFECT}

# state_detectors.py
TsumoPhaseDetector.consec_threshold = 2      # CNN ぶれ吸収
TsumoPhaseDetector.landed_consec = 2         # 着地確定
OjamaPhaseDetector.score_threshold = 70      # おじゃま 1 個分

# recognition_pipeline.py
CHAIN_HOLD_PER_STEP_SEC = 0.3                # CHAIN state ロック秒/連鎖段
MATCH_ACTIVE_HOLD_FRAMES = 10                # 試合 active hysteresis
CHAIN_BAN_FRAMES_AFTER_MATCH_START = 30      # 試合開始直後 CHAIN ban
PROB_BOARD_PUBLISH_ON_STABLE = True

# hybrid_classifier.py
DEFAULT_CNN_OVERRIDE_PROB = 0.75             # default
# pipeline では 0.90 に強化

# drift_detector.py
DEFAULT_DRIFT_CELL_THRESHOLD = 6
DEFAULT_DRIFT_FRAME_THRESHOLD = 3

# temporal_smoother.py
DEFAULT_WINDOW_SIZE = 15                     # 0.5s @ 30fps

# match_state.py
IN_MATCH_V_MAX = 170.0

# probabilistic_board.py
CERTAIN_THRESHOLD = 0.95

# online_hsv_calibrator.py
HIGH_CONF = 0.99
MIN_SAMPLES = 200
EMA_ALPHA = 0.05
RANGE_STD_MULT = 1.5

# score_ocr.py
NCC_MIN_CONFIDENCE = 0.55
NCC_AVG_MIN_CONFIDENCE = 0.65

# match_winner.py
DIGIT_DIFF_HAMMING = 10
DIGIT_ASYMMETRY_RATIO = 2.5

# score_zero.py
ZERO_NCC_THRESHOLD = 0.85

# win_panel.py
PANEL_NCC_THRESHOLD = 0.70

# ui_mask.py
DEFAULT_NCC_THRESHOLD = 0.75
```

---

(以上、`docs/IMAGE_RECOGNITION_OVERVIEW.md`)
