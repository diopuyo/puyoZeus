# 検出パイプライン総覧 (2026-04-30 時点)

ぷよぷよ eスポーツ動画から**フィールド / ネクスト / ダブルネクスト / 得点 / 予告お邪魔**を抽出する各サブシステムの構成、優先度、既知リスクを整理する。
集約点は `src/state_pipeline.py::StatePipeline.extract()` で、1 フレーム → `GameState` を返す。

---

## 0. 全体パイプライン (StatePipeline)

```
frame_bgr (1080p に正規化)
   │
   ├─[A] MatchEndDetector.update()        → match_end_locked (テンプレ NCC で やった!/ばたんきゅー)
   ├─[B] TelopDetector.is_visible()       → is_telop_visible
   │
   ├─[C] _extract_boards(frame)           ─┐
   │     PerVideoCalibrator.apply (BGR shift)
   │     ImageReader.read_both_boards     │  → board_p1, board_p2 (生)
   │       HybridClassifier (CNN v16 + HSV + UiMask)
   │       use_match_state / use_telop_mask
   │
   ├─[D] _extract_next(frame)              → next/dnext (NextDetector + StableNextDetector)
   ├─[E] _extract_score(frame)             → score / confidence (ScoreOcr 8桁 NCC)
   │
   ├─[F] お邪魔 pending 累積 (score 差分 → OjamaScoreInferrer)
   │
   ├─[G] 補正レイヤー (順次):
   │      bg_em (default OFF)
   │      EnhancedBoardTracker (V2.4: NextLink + Connectivity + Stateful)
   │      ScorePhysicsRefiner (default OFF)
   │      TemporalVotingRefiner (W11-D, window=3)
   │      ScoreBasedEraser (W12-B, chain 発火 4+ 強制 EM)
   │      PairLandingCheck (W13-A, 新着↔next_pair 拘束)
   │      ChainAnimationDetector (default OFF)
   │      PuyoStabilityRefiner (default OFF)
   │
   └─[H] ProbabilisticBoard 構築 (隠し段 + 予告お邪魔の確率分布)
        infer_hidden_row, infer_ojama_positions

→ GameState (board, next, dnext, score, pending_ojama, flags, pboard)
```

EVAL_INTERVAL_SEC = 0.6s (≈1.67 Hz)、BOARD_INTERVAL_SEC = 0.2s (5 Hz)。

---

## 1. フィールド (盤面 6×13、隠し段含む)

**最重要**。後段の評価指標・強化学習の入力として位置づけ。

| 段階 | 実装 | 役割 | 現状 |
|---|---|---|---|
| 観測 | `ImageReader.read_both_boards` | ROI (1P x=282, 2P x=1258, 384×720, セル 64×60) を切り出し | ROI 1080p ハードコード |
| 色分類 | `HybridClassifier` | CNN v16 主、HSV 補助、UI マスク (×印) で EMPTY 強制 | cross-video 93.85%、v18_m03 79.44% |
| 試合状態 | `MatchStateDetector` (HSV V 平均) | メニュー中は読み捨て | OK |
| テロップ | `TelopDetector` | 被覆セルを `COLOR_UNKNOWN` 化 | OK |
| Per-video キャリブ | `PerVideoCalibrator` | 試合開始の BG フレームから BGR シフトを推定して全フレームに適用 | sparse 評価 +3.81pt 効果 |
| 時系列フィルタ | `EnhancedBoardTracker` | 物理ルール違反棄却 + chain_event 検知 + V2.1/V2.3 統合 | sparse 評価では -7pt (連続 frame 前提) |
| 多数決 | `TemporalVotingRefiner` (window=3) | 過去 N フレーム多数決 | 連続 frame で動作 |
| Chain 整合 | `ScoreBasedEraser` | score 急増 → 4+ cluster を 5 frame 強制 EM | chain 発火時のみ動作 |
| ペア拘束 | `PairLandingCheck` | 新着 cell ↔ next_pair 一致を要求 (W14-A 修正済) | next_pair 検出失敗時は skip |
| 隠し段推論 | `hidden_row_inferrer.infer_hidden_row` | prev → cur の差分と prev next_pair から row 0 の確率分布 | 確率盤面に反映 |

### 既知の弱点
- **v18_m03 (最弱) 79.44%**。GRN/PUR/YEL の色混同が固定的に発生し、静的パッチ評価では改善せず → **温度的文脈** が必要 (連鎖タイミング、隣接列、輝度急変)。
- **v05_m55 58%**: sparse rendering or 配信 UI の差異が原因の可能性、未検証。
- **連鎖アニメ中の誤認**: `ChainAnimationDetector` は誤発火多く OFF、score-based eraser のみで対応 → score OCR が読めない瞬間は補正できない。
- **EnhancedBoardTracker は連続 frame 前提**: sparse 評価ハーネスでは害になり、生稼働では効果が出るはずだが定量検証は未完。

### 優先度: ★★★★★ (RL の入力として最重要)

---

## 2. ネクスト (Next pair) / ダブルネクスト (DNext pair)

**Phase W で安定化済み**。色 6 カテゴリ、1P/2P 両対応。

| 段階 | 実装 | 役割 |
|---|---|---|
| ROI | `next_detector.py` | 1P NEXT (710-785, 162-297)、DNEXT (765-815, 293-390)。2P は x=960 ミラー |
| 分類 | CNN (`patch_classifier`) + HSV ルール + Centroid 1-NN の **多数決** | 青背景 / L 字配置を考慮、INNER_CROP_RATIO=0.80 |
| 安定化 | `StableNextDetector` (window=2) | 2 連続 frame 同一のみ採用、揺れる瞬間は None |

### 既知の弱点
- **ROI が 1920×1080 ハードコード**。別大会・別 UI では再キャリブ必須。
- **NextPairClassifier (32×32 専用 CNN)** は実装済みだが、auto label が noisy で holdout 36% → 未活用。
- DNEXT は NEXT より小さく、回転中フレームで色取得失敗が起こり得る。

### 優先度: ★★★★ (PairLandingCheck・隠し段推論の前提条件)

---

## 3. 得点 (Score)

**8 桁 NCC OCR**。連鎖換算・予告お邪魔の起点として極めて重要。

| 段階 | 実装 | 役割 |
|---|---|---|
| ROI | `score_ocr.py` | 1P y=890-955 x=355-675、2P y=890-955 x=1253-1573、ピッチ 40px |
| 分類 | テンプレ `models/ui_templates/score_digits/digit_0..9.png` 50×40 BGR | 各桁 NCC 最大、`NCC_MIN_CONFIDENCE=0.55`、`NCC_AVG_MIN_CONFIDENCE=0.65`、`NCC_MARGIN_MIN=0.04` |
| フィルタ | StatePipeline 側で `confidence >= 0.5` のみ採用、`delta_max=50000` 超は連鎖アニメ中の不安定値として捨てる |

### 既知の弱点
- **連鎖発火中の "+1240" 計算式表示** が紛れ込む → 平均 NCC と 1-2 位 margin で抑制中だが完全ではない。
- **テロップ・全消し演出で OCR 失敗** → score 差分 = None 区間が発生し、その間の予告お邪魔が漏れる。
- **桁テンプレが 1080p 固定**。720p ネイティブ動画では Resize 経由でしか動かない。

### 優先度: ★★★★★ (お邪魔推論の根本入力、欠損は連鎖イベントの取りこぼしに直結)

---

## 4. 予告お邪魔ぷよ (Pending ojama)

**画像認識ではなく score 差分 → 公式式換算**で算出 (W1.1)。視覚予告 (上部のサイン) は未読み取り。

| 段階 | 実装 | 役割 |
|---|---|---|
| Score 差分 | StatePipeline `_update_ojama_pending` | t-1 → t の score 増分 (>0, ≤50000) を発火イベント化 |
| 換算 | `OjamaScoreInferrer.infer_from_score_delta` | `scoring.score_to_ojama` で公式式 + マージンタイム + leftover 端数 |
| 累積 | `_pending_ojama_p1`, `_pending_ojama_p2` | 1P 発火 → 2P pending に加算 |
| 位置推論 | `ojama_position_inferrer.infer_ojama_positions` | 画面内増加 M と予告 N の差分から、隠し段 (row 0) に積まれた候補列を確率分布化 |
| 量子盤面 | `ProbabilisticBoard` | 隠し段 OJM 確率を上書き |

### 既知の弱点
- **相殺 (offset) 未対応**: 双方が連続発火すると pending が二重加算される。実機ロジックでは相殺後に降る。
- **降下タイミング不明**: 「pending → 実際に落下」のイベントを score 差分だけからは特定できない。落下時に画面内 M が増えるが、検出が遅れる。
- **マージンタイム鯖違い**: 大会レギュレーション差で `OJAMA_RATE_STANDARD` が合わないリスク。
- **視覚予告 (相手側 上部のお邪魔予告アイコン)** はメモリ `project_ojama_inference_design.md` で「クロスチェック専用」位置付け、まだ未実装。
- **全消しボーナス持ち越し** はフラグ管理しているが、検出失敗 (score OCR 欠損) で破綻。

### 優先度: ★★★★ (RL の脅威評価に必須、ただし score OCR の信頼性に依存)

---

## 5. 補助状態 (試合終了 / テロップ)

| 検出器 | 仕組み | 用途 |
|---|---|---|
| `MatchEndDetector` | 「やった!」「ばたんきゅー」テンプレ NCC + 5s ロックダウン | 終了告知後の盤面更新を停止 |
| `TelopDetector` | 中央テロップ位置検出 + cells_covered_by_bbox | 被覆セル UNKNOWN 化 |
| `MatchStateDetector` | HSV V 平均で試合中 / メニュー判定 | ROI 領域全体の有効性 |

優先度: ★★ (state 補強)

---

## 6. 全体リスクマップ

| カテゴリ | リスク | 影響 | 対処状況 |
|---|---|---|---|
| **データ多様性** | v18_m03 のような UI で色混同が固定 | RL の入力ノイズが偏る | review labels 累計 3068 cells、追加が継続必要 |
| **CNN サチュレーション** | v7→v16 で +2pt のみ、自己強化やアーキ強化は頭打ち | これ以上 CNN 単独では伸びにくい | user-reviewed labels に投資中 |
| **連続 frame 評価未実施** | EnhancedBoardTracker・temporal_voting・score_eraser は本来連続 frame で効くが、定量評価は sparse のみ | 真の production 精度が不明 | 連続 frame で「N frame 連続誤検出」を可視化する harness が必要 (HANDOFF §9 で提案) |
| **ROI ハードコード** | 別大会動画では NEXT/score 全壊 | 配信動画ソース拡張時に詰む | per-video calibration は色のみ。座標 auto-detect は未実装 |
| **Score OCR 欠損** | 連鎖アニメ・テロップ中に 1〜数フレーム読めない → ojama pending 取りこぼし | 評価誤差・RL 報酬欠損 | confidence threshold + delta_max のみ。視覚予告クロスチェックは未実装 |
| **お邪魔相殺・降下未対応** | pending が累積し続ける、降下タイミング不明 | 実盤面と pending が乖離 | 視覚お邪魔予告 + 落下イベント検出が要追加 |
| **隠し段推論の暗黙仮定** | 1 列 1 個までの近似、複数 ojama 列に乗る場合は近似誤差 | 量子盤面の不正確 | 設計上の制限、現実頻度低 |
| **複数フレーム/動画の整合性** | v05_m55 が 58% など特定動画で著しく低い、原因未特定 | production 動画ごとに精度差が大きい | 動画別 holdout で要切り分け |
| **テスト網羅** | 733 → 996 まで増えたが、StatePipeline の連続 frame end-to-end テストは少ない | リファクタ時の regression 検出が弱い | 連続 frame harness と一緒に追加すべき |
| **依存ツール (ffmpeg)** | 不在環境で音声 mux 不可 | 配信合成動画の品質劣化 | `imageio-ffmpeg` バンドルで緩和済、`--no-audio` 強制でフォールバック |

---

## 7. 品質向上候補 (優先度順)

### S 優先 (最小工数で大効果)
1. **連続 frame 評価 harness** — render 時に「N frame 連続誤検出セル」と「v18_m03 等の hard 動画 5 秒区間の cell 別 confusion」を可視化。EnhancedBoardTracker / temporal_voting / score_eraser の真の効果を初めて定量化できる。HANDOFF §9 「video 19 のような hard 動画の連続 frame 評価」と同質。
2. **v18_m03 / v17_m11 / v19_m06 の追加 review** + CNN v17 訓練 (v16 init + 累計 review×N) — HANDOFF §9 即着手項目。期待 +0.5〜1pt cross-video, 弱点動画 +1〜2pt。

### A 優先 (中工数、構造強化)
3. **視覚お邪魔予告 + 落下イベント検出** — 上部の予告アイコン (4 段階くらい) を template / centroid で読む。score OCR 欠損のクロスチェックになり、相殺・降下の検出を可能にする (memory `project_ojama_inference_design.md` の「クロスチェック専用」位置付けを実装段階に進める)。
4. **Score OCR の 720p ネイティブ対応 + テロップ被覆フィルタ強化** — テロップ DOWN 区間の score 棄却を厳密化、削減した取りこぼしを ojama pending の精度に還元。
5. **ROI auto-calibration** — 試合開始の固定 UI 要素 (NEXT 枠の青枠線、score の 0 桁) から x/y オフセットを自動推定。別大会・別解像度動画の足切りが消える。

### B 優先 (中-長期、新機能)
6. **ChainAnimationDetector v2** — score+UI 両方の急変 (score 増 + 画面 motion 急増) を AND で取る。現状 motion 単独だと sparse 時に誤発火。
7. **Multi-scale CNN (24×24 = 3×3 cells)** — 隣接列の文脈を読ませる。v18_m03 の固定的色混同 (静的パッチでは判別不能) を温度的文脈で解く。
8. **NextPairClassifier の user-review** — 28K 自動ラベルを 1〜2K user review で fine-tune して holdout 36%→90% を狙う。NEXT 信頼性向上で PairLandingCheck の有効率が上がる。

### C 優先 (Phase X/RL 段階で回す)
9. **RL state extraction の整備** — `state_features.py` (1068dim) を最終化し、win_predictor を MLP → Transformer にスケール。
10. **OBS overlay の実機統合** — 優先度低、検出が安定してから。

---

## 8. 想定される議論ポイント (ロードマップ協議用)

1. **「画像認識卒業」のタイミング** — 現状の v16 ベースで RL に進むか、まだ認識精度を詰めるか。memory のフィードバックでは「強化学習優先・オーバーレイ後回し」「フィールド完全取得後の主目標は RL」 (2026-04-29) とあり、認識をどこまで完成と判定するか合意が要る。
2. **「v18_m03 79%」を許容するか** — 弱点動画の追加 review は手作業 200〜600 cells/動画で +1〜2pt。許容するなら CNN v16 確定で先へ進む。許容しないなら review 更に拡充で 1 セッション。
3. **視覚お邪魔予告の優先度** — score-only の現状から実装する場合、テンプレ収集 (v04..v19 から 4 段階アイコン × 19 動画) とテストで 1 セッション規模。RL 報酬の精度が直接効くので、RL 着手前に入れる価値が高い。
4. **連続 frame harness の優先度** — 補正レイヤー全体を「真の効果」で再評価できるため、CNN v17 を作る前にやる方が無駄が少ない (sparse 評価の罠を回避)。
5. **別大会動画の取り込み** — ROI auto-calibration 実装と引き換え。今のソース (3 配信者 + parallel 130 動画) で打ち止めにするか、新ソースを入れるか。
