# ぷよぷよ認識システム 現状と仕様 (2026-06-10時点)

> 対象コード: `src/`, `scripts/visualize_recognition.py`, `tests/test_ojama_accounting.py`
> 出典: `docs/PROJECT_STATE.md`, `docs/IMAGE_RECOGNITION_OVERVIEW.md`, `CLAUDE.md`, `src/ojama_accounting.py`, `src/scoring.py`, `src/board_state_machine.py`
> 確認できない事項は「未確認」と明記している。

---

## 1. 概要

ぷよぷよeスポーツ (1920x1080) の上級者対戦動画から、フレーム単位の画像認識で次の情報を読み取るシステム。

| 認識対象 | 出力 |
|---|---|
| 1P/2P 盤面 | 6列x13行 `numpy.ndarray` (color code) |
| 状態 (state) | STABLE / TSUMO_FALL / CHAIN / OJAMA_FALL / GRAVITY_SETTLE / MENU |
| score | score OCR (NCC、8桁) |
| next / dnext ペア | 上下ぷよ色 (HSV+CNN ハイブリッド) |
| 予告お邪魔 (会計) | score OCR 差分から連鎖終了時に一括計算 (2026-06-10実装) |
| 隠し段 (row 0) | 確率分布 (ProbabilisticBoard) |

### 盤面データ表現

- **サイズ**: 6列x13行 `numpy.ndarray`
- **行**: 0=最上段(隠し段/hidden row), 12=最下段
- **列**: 0=左端, 5=右端
- **色コード**:

| コード | 色 |
|---|---|
| 0 | 空 (EMPTY) |
| 1 | 赤 |
| 2 | 青 |
| 3 | 緑 |
| 4 | 黄 |
| 5 | 紫 |
| 9 | おじゃまぷよ |
| 10 | 不明 (COLOR_UNKNOWN) |

- **窒息判定**: 3列目 (index 2) の最上段 (row 0) にぷよがあれば DEAD
- **隠し段**: row 0 は `ProbabilisticBoard` で確率分布として保持。確定閾値 `CERTAIN_THRESHOLD=0.95`
- **可視領域**: row 1〜12 (12行)、row 0 は隠し段

---

## 2. 認識スタック

### 2.1 全体フロー

```
[1080p フレーム]
    → MatchStateDetector (HSV V平均で試合中/メニュー判定、IN_MATCH_V_MAX=170.0)
    → ImageReader.read_both_boards
         - ROI切出し (P1: x=282, y=160, w=384, h=720 / P2: x=1258, y=160, w=384, h=720)
         - BG fingerprint (空cell早期判定)
         - HybridClassifier (HSV+CNN, cnn_override_prob=0.90)
         - clear_floating_above_gap(min_gap=2)
         - _infer_hidden_rows
    → VideoChainTracker (puyo数急減で連鎖検出)
    → ScoreTracker (score OCR + 差分)
    → NextDetector (next/dnext)
    → BoardStateMachine.update (per side)
         - STABLE: confirmed_board更新 ★評価対象
         - 非STABLE: 前回STABLE盤面を凍結 (物理推論)
    → InferenceBoardGenerator / DriftDetector
    → SideResult (state, confirmed_board, prob_board, score, chain_event, ...)
```

### 2.2 state machine (BoardState)

`src/board_state_machine.py` で定義。**1サイドに1インスタンス**。

| State | 意味 | confirmed更新 | 採用ロジック |
|---|---|---|---|
| MENU | 試合外 (タイトル/リザルト) | クリア | MatchStateDetector 失敗時 |
| STABLE | 平常時 ★評価対象 | 初回のみ多数決、以降は state遷移時物理推論 | CNN連続N=6 frame一致 |
| TSUMO_FALL | ツモ落下中 | 凍結 | puyo数+1〜+2が連続2 frame |
| CHAIN | 連鎖中 | 凍結 | VideoChainTracker event |
| OJAMA_FALL | おじゃま落下中 | 凍結 | 相手score_delta≥70 |
| EFFECT | 全消し演出等 | 凍結 | 現状 skeleton (常にNone) |
| GRAVITY_SETTLE | 連鎖終了直後の重力settle | 凍結 | CHAIN→GRAVITY_SETTLE遷移 (2026-06-06 default ON) |

**重要原則**: 指標評価は「両者STABLE時の`confirmed_board`のみ」で実行。非STABLE中は前回STABLE盤面を凍結 (`feedback_chain_phase_physics_only`)。

主要パラメータ:

| パラメータ | 値 | 意味 |
|---|---|---|
| `stable_frame_count` | 6 (pipeline default) | STABLE確定の連続frame数 |
| `STABLE_WARMUP_FRAMES` | 12 | STABLE遷移直後のconfirmed凍結frame数 |
| `consec_threshold` | 2 | TSUMO進入の連続frame数 |
| `GRAVITY_SETTLE_MIN_FRAMES` | 8 | GRAVITY_SETTLE最短保持frame数 |
| `GRAVITY_SETTLE_MAX_SEC` | 1.5 | GRAVITY_SETTLE最大保持秒 |
| `cnn_override_prob` | 0.90 (pipeline) | CNN>HSVの確信度閾値 |
| `DRIFT_CELL_THRESHOLD` | 6 | 1 frameの不一致cell数 |
| `DRIFT_FRAME_THRESHOLD` | 3 | 連続drift frameでresync |

### 2.3 HybridClassifier (HSV + CNN)

`src/patch_classifier.py` / `src/hybrid_classifier.py`。

- UI mask hit → EMPTY強制
- CNN確信度 ≥ `cnn_override_prob` (pipeline default=0.90) → CNN採用
- 不確信 + HSV一致 → CNN
- 不確信 + HSV不一致 → **HSV採用** (決定論性優先)

HSV strong color override: 赤 H 0-10 / H 165-180、紫 H 125-160 は HSVで強制上書き。

### 2.4 自動HSV較正 (OnlineHsvCalibrator)

`src/online_hsv_calibrator.py`。

- CNN高確信(≥0.99) + HSV単独判定一致 + 連続STABLE条件で信頼サンプル抽出
- 色別EMA (alpha=0.05) で平均/分散更新
- N≥200サンプルで動画別ranges採用 (mean±1.5σ)
- `ImageReader.set_color_ranges_from_simple` でclassifierに注入

**2026-06-10時点のステータス**: 自動HSVのみで99.5%以上達成済み (手調整per-video HSV廃止可)。`--no-per-video-hsv` フラグで手調整 inject をスキップし自動HSVのみで動作確認可。

### 2.5 背景FP (Background Fingerprint)

- 試合開始+5 frame以降、puyo 0個盤面5枚で `capture_robust_fingerprint`
- cell単位のHSV距離で空cell早期判定 (tier1閾値: `BG_EXTREME_THRESHOLD_DEFAULT`)
- OJAMA_FALL→STABLE専用warmup: `OJAMA_TIER1_WARMUP_FRAMES=8` frame間tier1をskip (v70列崩壊対策、default ON)

### 2.6 連鎖timing検出

| 機能 | 実装 | default |
|---|---|---|
| 機能D: 掛け算式検知 (連鎖開始) | score ROI OCR=None + ink_ratio > 閾値 + `CHAIN_FORMULA_CONSEC_FRAMES`連続 | ON |
| VideoChainTracker (puyo急減) | `ERASURE_MIN_DROP=4` puyo数急減で発火 | ON |
| game-event連鎖終了 | 次ツモ変化 or 連鎖側ojama降下で CHAIN終了 | ON |
| 案X: NextSlide CHAIN即終了 | 次ツモスライド signal で即CHAIN終了 | OFF |
| 機能B: score急増早期発火 | 自side score_delta≥80 で即CHAIN | OFF |
| GRAVITY_SETTLE状態 | 連鎖終了直後の重力settle window (CHAIN→GRAVITY_SETTLE→STABLE) | ON |
| 案γ: slide ojama-hold上書き | CHAIN中nextスライドでojama保留を強制終了 | ON |

### 2.7 GRAVITY_SETTLE

`src/board_state_machine.py` 定数 (2026-06-06採用、default ON)。

- `GRAVITY_SETTLE_MIN_FRAMES=8` (≈0.27s @30fps): ぷよ数変化が安定するまでの最短待機
- `GRAVITY_SETTLE_MAX_SEC=1.5`: タイムアウトで強制STABLE復帰
- `GRAVITY_SETTLE_PUYO_DIFF_THRESHOLD=2`: 連続frame間ぷよ数差がこれ以内なら静止と判定
- お邪魔会計: `(CHAIN or GRAVITY_SETTLE)→STABLE` 遷移をトリガーに連鎖終了イベントを発火

### 2.8 予告発光ガード (glow v5)

- `--ojama-warning-glow-guard` オプション: 相手連鎖の予告おじゃま演出による盤面上部多色発光 (V_high_ratio) を検知し、STABLE中のconfirmed_boardをfrozen_boardで保護
- 黄ぷよ→おじゃまの誤認を防ぐ
- **ライブラリ default=False (OFF)** (フラグ明示が必要)

### 2.9 隠し段推論 (HiddenRowInferrer → ProbabilisticBoard)

`src/hidden_row_inferrer.py`。

- prev_board → cur_boardの差分とネクスト履歴から row 0の確率分布を推論
- n=2 (新規2セル、ペア整合): 隠し段はEMPTY確定
- n=1: もう1セルが隠し段にある → 列候補で確率分布
- n=0/3+: 推論しない
- TSUMO_FALL→STABLE着地時に `infer_hidden_row` を呼び出し `prob_board` に格納

### 2.10 各種refiner (default ON/OFF一覧)

| 機能 | default | フラグ |
|---|---|---|
| T2高確信yield (infer_placement色破壊修正) | ON | `--t2-highconf-yield` / `--no-t2-highconf-yield` |
| infer-empty-guard (hallucination防止) | ON | `--infer-empty-guard` / `--no-infer-empty-guard` |
| stable-recovery-gate (事後復旧ゲート) | ON | `--stable-recovery-gate` / `--no-stable-recovery-gate` |
| red-hue-wrap-fix (赤色相折り返し補正) | ON | `--red-hue-wrap-fix` / `--no-red-hue-wrap-fix` |
| specular-robust-saturation (光沢ハイライト除外) | ON | `--specular-robust-saturation` / `--no-specular-robust-saturation` |
| ojama-tier1-warmup (OJAMA専用tier1 warmup) | ON | `--ojama-tier1-warmup` / `--no-ojama-tier1-warmup` |
| ojama-visual-detection (フェーズA4) | ON | `--enable-ojama-visual-detection` |
| ojama-visual-chain-exit (フェーズA4) | ON | `--enable-ojama-visual-chain-exit` |
| constraint-fill (NEXT累積制約) | OFF | `--constraint-fill` |
| chain-exit-warmup (CHAIN→STABLE凍結) | OFF | `--chain-exit-warmup` |
| chain-exit-next-signal (案X) | OFF | `--chain-exit-next-signal` |
| ojama-warning-glow-guard (glow v5) | OFF | `--ojama-warning-glow-guard` |
| landing-color-fix (着地色修正案1) | OFF | `--landing-color-fix` |

---

## 3. 達成状況 (Phase I 認識精度)

### 3.1 確定済み数値

| 指標 | 数値 | 条件 | 出典 |
|---|---|---|---|
| STABLE cell-level acc (手調整+自動HSV合算) | 99.87% | `--hsv-state` per-video inject使用 | `memory/project_phase_i_metric_actual_state.md` |
| **自動HSVのみ** (per-video手調整廃止可) | **99.7%+** | `--no-per-video-hsv` (OnlineHsvCalibrator動作) | `memory/project_session_2026-06-06_handoff.md` |
| 汎用化目標 | 99.5%以上 | 自動HSVのみ、手調整なし | `memory/project_generalization_target_auto_hsv.md` |
| CNN cell holdout (Phase B v1) | 99.61% | `cnn_phase_b_v1.pt` | `docs/IMAGE_RECOGNITION_OVERVIEW.md` |

**重要**: `verdict=PASS` の判断は数値だけで行わない。Phase I は数値上99.85%+を達成したがv70列崩壊 fail-silent (連鎖後全EMPTY崩壊) の存在により `verdict=FAIL` とされた経緯あり (`memory/project_phase_i_metric_actual_state.md`)。

### 3.2 Phase I 実質達成事項 (2026-06-06時点)

- 機能D (掛け算式連鎖開始検知): 採用済み (PR#11/12 merged)
- 自動HSVのみで99.7%以上: 確認済み (手調整廃止可)
- GRAVITY_SETTLE状態: 採用済み (PR#13、不具合A本質修正)
- glow v5 予告発光ガード: 採用済み (PR#11/12 merged)

### 3.3 学習有利不利判定の達成値

| モデル | 指標 | 値 |
|---|---|---|
| LR video holdout | H1 66動画混在 | 0.694 |
| LR LOOV avg | H4.1 MLP top_20 | 0.762 (最高) |
| LR video holdout | Top-tier H1 (39動画) | 0.690 |
| end phase | Top-tier H1 | 0.964 (最高) |

---

## 4. お邪魔会計モデル (2026-06-10 新規実装)

`src/ojama_accounting.py` / `tests/test_ojama_accounting.py` (53テスト)

### 4.1 ゲームルール実装

| ルール | 実装内容 |
|---|---|
| 生成タイミング | 連鎖終了時に一括。連鎖中は生成しない |
| 生成量計算 | `(chain_total_score + leftover) ÷ rate`、端数は次連鎖に繰越 |
| 基準rate | `OJAMA_RATE_STANDARD=70` (src/scoring.py) |
| 相殺の向き | 自分が連鎖を撃つと「自分への予告 (incoming)」を打ち消す → 余剰 (surplus) を相手へ |
| 全消し | 得点計算側 (ALL_CLEAR_BONUS=2100) の責務。`all_clear_pending_p1/p2` は廃止・常False |
| 落下 | `on_tsumo_settled` 呼び出しで最大 `THEORY_DROP_PER_TURN=30` 個/ターンをdrain |
| マージンタイム | `MARGIN_TIME_START_SEC=96.0` 秒以降、`MARGIN_TIME_DECAY_INTERVAL_SEC=16.0` 秒ごとに rate × `MARGIN_TIME_DECAY_FACTOR=0.75` 減衰 (src/scoring.py)。**試合相対経過秒で計算** (2026-06-10修正、クリップ先頭経過秒との混同バグを根治) |

### 4.2 定数一覧

| 定数 | 値 | 意味 |
|---|---|---|
| `ON_FIELD_CAP` | 72 | 可視フィールド全セル数 (12行×6列) |
| `PENDING_HARD_CAP` | 72 | forecast有界上限 (=ON_FIELD_CAP) |
| `PENDING_ABS_CAP` | 216 | forecast絶対サニティ上限 (3画面分) |
| `CHAIN_COALESCE_WINDOW_SEC` | 2.5 | state明滅を同一連鎖とみなす時間窓 |
| `CHAIN_END_PENDING_TIMEOUT_FRAMES` | 30 | score None継続タイムアウト (≈1秒@30fps) |
| `SCORE_RESET_THRESHOLD` | 500 | 試合境界検知用score減少閾値 |
| `THEORY_DROP_PER_TURN` | 30 (=OJAMA_MAX_DROP_PER_TURN) | 1ターン最大落下個数 |
| `CONFIDENCE_SCORE_OCR_ONLY` | 0.85 | snapshotのconfidence固定値 |
| `CHAIN_TOTAL_SANITY_MAX` | 200,000 | 連鎖合計スコアのサニティ上限 |
| `MENU_RESET_CONSEC_FRAMES` | 3 | MENU連続frameでリセット |

### 4.3 API

**主要クラス**:

- `OjamaAccountingTracker`: 試合1本分の状態保持wrapper
  - `reset(match_start_sec=None)`: 全帳簿クリア
  - `on_state_transition(side, prev_state, curr_state, score, t_sec)`: BoardState遷移通知 (連鎖終了イベント駆動)
  - `on_tsumo_settled(side, t_sec)`: TSUMO着地時の予告drain
  - `get_snapshot(t_sec) -> OjamaAccountSnapshot`: 現在状態のスナップショット取得

- `OjamaAccountSnapshot` (frozen dataclass): 1時刻の両者お邪魔会計スナップショット

**主要フィールド**:

| フィールド | 意味 |
|---|---|
| `pending_p1` / `forecast_p1` | 1Pが受ける予告お邪魔個数 (エイリアス) |
| `pending_p2` / `forecast_p2` | 2Pが受ける予告お邪魔個数 (エイリアス) |
| `pending_p1_capped` / `pending_p2_capped` | ON_FIELD_CAP (72) 有界 |
| `net_balance_capped` | `pending_p2_capped - pending_p1_capped` (正=1P有利、範囲 -72〜+72) |
| `net_ojama_balance` | `pending_p2 - pending_p1` (有界なし) |
| `offboard_p1` / `offboard_p2` | 画面外あふれ推定個数 (forecast - ON_FIELD_CAP) |
| `total_generated_by_p1/p2` | 累積生成量 (相殺前) |
| `total_offset_by_p1/p2` | 累積相殺量 |
| `total_dropped_to_p1/p2` | tsumo着地drainの累積 |
| `leftover_p1` / `leftover_p2` | score換算端数繰越 |
| `chain_total_score_p1/p2` | 最後の連鎖合計得点 (検証用) |
| `chain_end_triggered_p1/p2` | 今フレームに連鎖終了イベントが立ったか |
| `score_at_chain_start_p1/p2` | 連鎖開始直前score (検証用) |
| `confidence` | 固定値 0.85 (CONFIDENCE_SCORE_OCR_ONLY) |
| `overflow_risk_p1/p2` | forecast ≥ overflow_threshold |
| `all_clear_pending_p1/p2` | 廃止、常False |

**net_balance_cappedの正規化**: `(net_balance_capped + 72) / 144` で 0〜1 に正規化可能 (tests/test_ojama_accounting.py で検証済み)。

### 4.4 連鎖終了検知の詳細

- 連鎖開始: `prev_state ∉ {CHAIN}` かつ `curr_state == CHAIN` → `score_at_chain_start = last_valid_score` に設定
- 連鎖開始フレームでscore=Noneの場合、直前STABLEの`last_valid_score`を使用 (掛け算式表示で OCR失敗する実機の動作に対応)
- 連鎖終了: `(CHAIN or GRAVITY_SETTLE) → STABLE` 遷移 + `chain_active=True`
- score None時: `chain_end_pending=True` にして後続フレームで遅延確定。タイムアウト (30f) で破棄
- state明滅デバウンス: finalize後 `CHAIN_COALESCE_WINDOW_SEC=2.5` 秒以内の再CHAIN開始は `score_at_chain_start` を上書きしない (1連鎖=1 finalizeを保証)
- 試合境界: score減少≥500 or MENU遷移で帳簿全リセット + `_match_start_sec` 更新

### 4.5 後方互換API

旧 `visualize_recognition.py` 呼び出しのために維持 (新実装ではno-opまたはルーティング):

- `update_from_score(...)`: `on_state_transition`/`on_tsumo_settled` に内部ルーティング
- `update_from_boards(...)`: no-op (visible_ojama保存のみ)
- `update_accounting_with_chain(...)`: snapshotを返すだけ

### 4.6 限界・未完事項

- **算出予告 vs 画面の予告アイコンの目視突合は未完**: 画面ROI(予告アイコン領域)の特定が未実施
- **視覚予告アイコン検出は不使用**: 誤読の温床であるため `CONFIDENCE_VISUAL_AGREE` / `CONFIDENCE_VISUAL_MISMATCH_PENALTY` は定義済みだが現在未使用
- **state_pipeline経由の統合は未完**: `recognition_pipeline.py` の `SideResult` に `per_side BoardState` へのアクセス経路が整備されていないため、会計トラッカーはviz経由でのみ駆動

---

## 5. レビュー可視化 (scripts/visualize_recognition.py)

### 5.1 overlay要素

各サイドのROI上に以下を描画:

| 描画内容 | 場所 | 詳細 |
|---|---|---|
| 状態 (state) ラベル + 枠線 | ROI上端 + 枠 | 色分け: STABLE=緑, CHAIN=ピンク, TSUMO=橙, OJAMA=シアン, GRAVITY_SETTLE=橙黄, MENU=グレー |
| ぷよ色シンボル (R/B/G/Y/P/O/?) | 各cell中央 | 可視12行のみ。EMPTY cellは非表示 |
| score | 状態ラベル横 | score>0の場合のみ |
| next / dnext | ROI下端 | `N:XX D:XX` 形式 (2文字ずつ色記号) |
| お邪魔会計 (3行) | ROI下端の下 | 行1: `pend:N net(off後):±K c:0.xx` / 行2: `drop:x off:Y` / 行3: `OB+N(approx)` |
| 隠し段確率 (帯) | ROI上端より上 (専用帯52px) | HIDDEN_ROW_MIN_PROB=0.10 以上のセルのみ。確率に応じてフォントサイズ変化 |
| 画面外お邪魔 (OB) | 隠し段帯右端 | offboard_ojama > 0 の場合に赤字で `O+N` |
| グローバル情報 | 画面上部 | frame_idx / t_sec / 1P-2P state + score |

### 5.2 主要オプション

| オプション | 意味 |
|---|---|
| `--video PATH` | 入力動画パス (必須) |
| `--output PATH` | 出力動画パス (必須) |
| `--no-per-video-hsv` | per-video手調整HSV inject をスキップ。OnlineHsvCalibrator (自動HSV) は引き続き動作。**汎用精度確認用** |
| `--dump-ojama-accounting PATH` | OjamaAccountingTrackerの各フレームsnapshotをJSONL保存 |
| `--dump-board-log PATH` | confirmed_board / state / chain_event をJSONL保存 (強化アナリスト用) |
| `--dump-board-log-detailed PATH` | raw_cnn_board / raw_hsv_board / bg_fp_distance_grid / constraint_fill_changed_cells 等の詳細JSONL保存 |
| `--sample-interval FLOAT` | 認識処理frame間隔 (秒)。default=0.033 (30fps) |
| `--max-sec FLOAT` | 処理最大秒数 (0=全部) |
| `--gravity-settle-state` / `--no-gravity-settle-state` | GRAVITY_SETTLE状態の有効/無効 (default=有効) |

### 5.3 レビュー動画の作り方

```bash
# 基本 (per-video HSV inject あり、自動解決)
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video data/evaluation_videos/TARGET.mp4 \
    --output data/evaluation_videos/TARGET_viz.mp4

# 自動HSVのみ (汎用精度確認)
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video data/evaluation_videos/TARGET.mp4 \
    --output data/evaluation_videos/TARGET_viz_autohsv.mp4 \
    --no-per-video-hsv

# お邪魔会計JSONL付き
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video data/evaluation_videos/TARGET.mp4 \
    --output data/evaluation_videos/TARGET_viz.mp4 \
    --dump-ojama-accounting data/evaluation_videos/TARGET_ojama.jsonl

# board_log詳細付き (真因診断)
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video data/evaluation_videos/TARGET.mp4 \
    --output data/evaluation_videos/TARGET_viz.mp4 \
    --dump-board-log-detailed data/evaluation_videos/TARGET_boardlog.jsonl
```

> 長時間生成のプロセス管理メモ (2026-06-10セッション知見): `setsid -f` でのdetach起動は起動元セッション終了時にreapされ未finalize破損する事例があった。Claude harness管理のバックグラウンド実行(単一プロセス・寿命保持)の方が確実。生成後は必ず `cv2.VideoCapture` で frames>0 を確認する (サイズだけ見ると破損動画を見落とす)。

---

## 6. 既知の限界・残課題

### 6.1 お邪魔会計

| 課題 | 詳細 |
|---|---|
| 予告 vs 画面突合の未完 | 画面の予告アイコン(大中小菱形アイコン列)のROI特定が未実施。算出値と画面表示の自動突合ができない |
| 視覚予告アイコン検出の不使用 | 誤読の温床のため意図的に不使用。score OCR差分一本化 |
| state_pipeline再統合の未完 | `OjamaAccountingTracker.on_state_transition` は `visualize_recognition.py` 経由で駆動しており、`recognition_pipeline.py` への統合は未着手 |
| OJAMA_FALL中の相殺タイミング | 実際のぷよぷよでは連鎖中(お邪魔降下前)に相殺が確定するが、現実装の連鎖終了トリガーのタイミング精度はstate machine依存 |

### 6.2 認識全般

| 課題 | 詳細 |
|---|---|
| 序盤(+20s以内)の誤認 | BG FP採取前はデフォルトHSVのみ。ネクスト履歴未蓄積でCNN fallback |
| 2 frame連続同一誤認 | drift検出閾値(6cell×3frame)未満の微小drift |
| EffectPhaseDetector未実装 | 全消し演出/連鎖カットインのskeletonのみ(常にNone) |
| OnlineHsvCalibratorのpipeline未統合 | モジュール実装済みだがRecognitionPipeline本体への統合が未完 |
| ネクストROIの動画依存 | 1920x1080 v10確定ROI。別大会UIでは再キャリブ必要 |
| 光沢ぷよの誤EMPTY | 赤ぷよ表面の白ハイライト (S_min未達→EMPTY誤判定)。specular-robust-saturation (default ON) で軽減済みだが完全解消ではない |

### 6.3 測定・評価

| 課題 | 詳細 |
|---|---|
| 3者合意evalのfail-silent | CNN==HSV一致なのにconfirmed違う後処理破壊を99.86%が隠蔽した前例あり (`feedback_consensus_eval_fail_silent`) |
| 試合切り出しvs raw動画の精度差 | evaluation_videosは試合切り出し済。raw動画は試合外フレームを含み精度数値が下がる |

---

## 7. 使い方 (主要コマンド)

### 7.1 認識可視化動画

```bash
# 基本
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video data/evaluation_videos/v40_match7_125s.mp4 \
    --output data/evaluation_videos/v40_viz.mp4

# 最大60秒のみ処理
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video data/evaluation_videos/v40_match7_125s.mp4 \
    --output data/evaluation_videos/v40_viz.mp4 \
    --max-sec 60
```

### 7.2 board_log / ojama-accounting dump

```bash
# board_log (confirmed_board + state を JSONL)
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video data/evaluation_videos/v40_match7_125s.mp4 \
    --output data/evaluation_videos/v40_viz.mp4 \
    --dump-board-log data/evaluation_videos/v40_board_log.jsonl

# お邪魔会計 JSONL
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video data/evaluation_videos/v40_match7_125s.mp4 \
    --output data/evaluation_videos/v40_viz.mp4 \
    --dump-ojama-accounting data/evaluation_videos/v40_ojama.jsonl
```

### 7.3 テスト実行

```bash
# 全テスト (1,400+)
python -m pytest tests/ -v

# お邪魔会計テストのみ (53テスト)
python -m pytest tests/test_ojama_accounting.py -v

# 認識pipeline テスト
python -m pytest tests/test_recognition_pipeline.py tests/test_board_state_machine.py -v
```

> 高速化メモ: pytest-xdist (`-n auto`) はこのテスト群では torch/CNN のワーカー毎import overhead が並列利得を上回り**逆に遅くなる** (実測: 単独717s < -n auto 959s) ため非推奨。GPU並列もテストはCPUバウンドのため無効。

### 7.4 認識精度評価 (強化アナリスト)

```bash
# STABLE cell-level acc 測定
PYTHONPATH=. ./venv/bin/python -m scripts.measure_stable_cell_acc \
    --workers 4 \
    --videos data/evaluation_videos/
```

---

## 8. 参照

| ドキュメント | 内容 |
|---|---|
| `docs/IMAGE_RECOGNITION_OVERVIEW.md` | 認識スタック完全説明 (1,039行) |
| `docs/PROJECT_STATE.md` | ディレクトリ構成、Phase進捗、学習結果累積 |
| `docs/INDICATOR_REFERENCE.md` | 45指標フル定義 |
| `docs/INDICATOR_ROADMAP.md` | Phase H1〜L 詳細ロードマップ |
| `docs/CYCLE_FINDINGS.md` | cycle検証で確定したルール |
| `data/verify/learning_impact_audit.md` | 残課題リスト |
| `src/ojama_accounting.py` | お邪魔会計実装 |
| `src/scoring.py` | 得点計算・マージンタイム定数 (`MARGIN_TIME_START_SEC=96.0`, `OJAMA_RATE_STANDARD=70`) |
| `src/board_state_machine.py` | state machine定数 (GRAVITY_SETTLE系含む) |
| `scripts/visualize_recognition.py` | レビュー可視化・全オプション定義 |
| `memory/MEMORY.md` | セッション間引継ぎ (project_cycle_*, feedback_*) |
| `memory/project_session_2026-06-06_handoff.md` | Phase I達成状況の直近引継ぎ |
| `memory/project_generalization_target_auto_hsv.md` | 自動HSVのみ99.5%必須の根拠 |
| `memory/feedback_viz_eval_required.md` | viz目視併用必須のルール |
| `memory/feedback_consensus_eval_fail_silent.md` | 3者合意evalのfail-silent警告 |
