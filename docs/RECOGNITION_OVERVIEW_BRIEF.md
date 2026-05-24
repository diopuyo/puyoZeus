# ぷよ認識システム 概要 (2026-05-15 時点)

## 全体構成

```
動画 frame (1920x1080)
   ↓
[ImageReader]
   - 各 cell BGR patch 切り出し (1P × 72 cell, 2P × 72 cell)
   - 背景 FP (= 試合開始時の空セル HSV) で早期 EMPTY 判定
   - HybridClassifier (HSV + CNN) で色分類
   - UI Mask / TelopMask で X 印・テロップ被覆 cell を UNKNOWN
   ↓ cnn_board (CNN 認識生盤面)
[RecognitionPipeline]
   - MatchStateDetector: 試合中/外
   - VideoChainTracker: 連鎖発火検出 (puyo 減少 + ChainSimulator)
   - ScoreOcr: 1P/2P score 読取
   - NextDetector: NEXT/dnext pair 読取
   - NextSlideDetector: NEXT スライド motion 検出
   - BoardStateMachine (per side): MENU/STABLE/TSUMO_FALL/CHAIN/OJAMA_FALL/EFFECT
   ↓ confirmed_board (STABLE 確定盤面)
```

## CNN モデル

- `models/cnn_phase_b_large_v3.pt` (default)
- アーキ: 4 層 Conv + BN + Dropout (= CnnPatchClassifierLarge)
- 入力: 30×32 BGR+HSV 6 channel patch
- 出力: 7 クラス (EMPTY / RED / BLUE / GREEN / YELLOW / PURPLE / OJAMA)
- 訓練: 10 動画 31K cells、 val accuracy **99.55%**
- 学習動画: test_v50, v50_match1, v89_match1, v91_match1, v29_match2, v40_match7, v57_match2, v51_match2, v70_match2, v89_match3

## HybridClassifier 判定ロジック

```
cnn_prob >= 0.70  →  CNN 採用
cnn_prob <  0.70 + CNN/HSV 一致  →  両者一致 (CNN)
cnn_prob <  0.70 + CNN/HSV 不一致 + prob < 0  →  UNKNOWN (現状 disabled)
cnn_prob <  0.70 + 不一致 + prob >= 0  →  HSV 採用
```

## State Machine (per side, 1P/2P 独立)

| State | 意味 | CNN 採用 | confirmed 更新 |
|---|---|---|---|
| MENU | 試合外 | × | リセット |
| STABLE | 平常時 | △ (= 多数決) | state 遷移時のみ |
| TSUMO_FALL | ツモ落下中 | × | しない (= physics) |
| CHAIN | 連鎖中 | × | しない |
| OJAMA_FALL | お邪魔落下 | × | しない |
| EFFECT | 演出中 | × | しない |

## 多層補正パイプライン (Phase-by-Phase)

着地後 0.4-0.6 秒で正しい色に収束させる:

1. **landing_grace** (= 12 frame = 0.2s @ 60fps): physics-inferred final_board を hold
2. **landing_vote** (= 24 frame = 0.4s): CNN 観測の多数決 + NEXT 色 HSV votes で cell 色更新
3. **long-term vote** (= 18 frame history, 75% ratio): STABLE 中の不一致 cell を CNN 多数決で override (浮きぷよ防止 + post-gravity sweep)
4. **prev_stable 補完**: UNKNOWN cell を直前 STABLE 値で埋める (cycle 71j)
5. **CNN 観測補完**: UNKNOWN cell を現 cnn_board で埋める (F1)
6. **NEXT 履歴整合性**: ever_seen 集合外の色を UNKNOWN/HSV-replace (F1-B, Innovation D)

## 物理推論 (`infer_placement`)

TSUMO_FALL → STABLE 遷移時:
1. `enumerate_landing_patterns(prev_confirmed)` で物理的可能着地位置を全列挙
2. CNN diff cells (= 空→puyo になった cells) で **position filter** (cycle 71v)
3. NEXT pair 色 を pattern.cells に割当
4. 着地 2 cell の HSV と NEXT 色 HSV center 距離で順序確定 (cycle 71l β2')
5. `resolve_after_placement` で即時連鎖判定 (chain_sim), 連鎖あれば final_board に上書き
6. 全消し時は `_chain_until` を延長して overlay 期間カバー

## state 遷移検出

- TSUMO_FALL ← cnn_board が baseline + 1〜2 puyo 連続観測
- TSUMO_FALL → STABLE ← `diff >= min_increase` で `landed_consec` 連続同一 (= 33ms)
   OR `slide_motion=True` (= NEXT スライド検出) で即時
- CHAIN ← chain_tracker が puyo 数減少を観測
- CHAIN → STABLE ← chain_until 経過

## 背景 FP (BackgroundFingerprint)

- 試合開始 5 frame 後の空盤面 5 枚から各 cell HSV を median 集約
- 推論時、 cell HSV が背景 FP から `DEFAULT_EMPTY_HSV_DISTANCE` (= 35) 未満なら早期 EMPTY 判定
- CNN にかける前段フィルタで背景 → puyo 誤認を抑止
- `infer_placement` でも bg_fp 一致 cell を物理推論の diff から除外 (= ghost commit 防止)

## OnlineHsvCalibrator (動画別 HSV 自動学習)

- STABLE 中の信頼サンプルから動画別 HSV 範囲を抽出
- 200 samples 以上溜まった色から **段階的 inject** (cycle 71v): B/P (青/紫) は 1.5-4s で適用、 codec ズレ対策

## NEXT 履歴 ever_seen (F1-B)

- NEXT pair の cap 8 でスクロールアウトしても、 試合中観測した色は永続記録
- _validate_next_history で「履歴外色 = 誤認」 判定が消えない問題を解消

## 主要パラメータ (cycle_5 値)

| Param | 値 | 説明 |
|---|---|---|
| `STABLE_CNN_HISTORY_FRAMES` | 18 | long-term vote 履歴 |
| `STABLE_OVERRIDE_MIN_RATIO` | 0.75 | override 発火閾値 |
| `LANDING_VOTE_FRAMES` | 24 | 着地後 vote 完了時間 |
| `LANDING_VOTE_MIN_RATIO` | 0.3 | 着地 vote 確定閾値 |
| `LANDING_GRACE_FRAMES` | 12 | physics hold 期間 (0.2s @60fps) |
| `DEFAULT_EMPTY_HSV_DISTANCE` | 35 | bg FP 距離閾値 |
| `MATCH_JUST_STARTED_WINDOW_FRAMES` | 60 | 試合開始空フィールド強制 window |
| `ALL_CLEAR_OVERLAY_HOLD_SEC` | 1.5 | 全消し chain hold 延長 |

## 既存研究との関係

- mbrown/puyogg/puyoai: 色のテンプレート match や HSV のみ → 認識精度に天井あり
- 本プロジェクト: CNN を主軸に多層補正 + 物理推論 + 動画別 HSV 学習 → 99%+ を目指す

詳細: `docs/IMAGE_RECOGNITION_OVERVIEW.md` (1039 行)
