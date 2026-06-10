# 最終ロードマップ: 上級者ぷよぷよ有利不利判定 + リアルタイムオーバーレイツール

**作成**: 2026-05-08
**改訂**: 2026-05-08 (戦略概念訂正 + 最終目標追加)

## 最終目標

ぷよぷよeスポーツ (上級者対戦) のリアルタイム有利不利判定ツール:
- 配信オーバーレイ (OBS ブラウザソース) として動作
- 上級者の戦術パターンを学習データから発見
- 説明可能性を維持しつつ、人間想定を超える戦略評価
- 視聴者・解説者の補助 / コーチング応用

## 設計方針

### 戦略概念訂正 (web 検索 + ユーザー指摘より)
- ojama 降下中はぷよ操作不能 (停止)
- カウンター = ojama 受けて高い発火点で発火、ojama 消化 + 大威力
- 上部 = 配置リズム速 (落下距離短縮)、上級者は advantage、中級者は penalty
- 連鎖時間 = 落下マス + クイック有無で変動 (本作の正確値は学習で推定)
- 形比率: 8 割が先折り GTR (推定、学習で検証)
- 全消しボーナス: 2,100点 = ojama 30個 = 5段、即発生せず次消去で発生

### 指標設計の中心思想

#### 思想 1: 観測軸を提供 → 学習で重要度を発見
- 私が「これが重要」と決めない
- 戦略概念をデータ観測指標に翻訳
- 重要度・閾値は学習結果から ranking
- ablation で各指標の真の貢献を特定
- Phase Aware で序盤・中盤・終盤の重要度差を観察

#### 思想 2: 形は手段、機能が本質 (★ 重要追加)
**GTR / サブマリン / 階段 等の「形」分類は二次的、本質は「機能・能力」**

理由: 火力が出て、中盤の厚みがあるなら、形が GTR でもサブマリンでも結果は変わらない。
形は手段であって目的ではない。指標は「達成された機能」を測るべき。

**機能能力指標 (Capability Indicators) を一次軸とする**:
- ready_chain_count: 現連鎖数 (即発火可能な連鎖数)
- ignition_distance: 発火距離 (何手で発火可能か)
- current_fire_power: 現威力 (即発火した場合の ojama 数)
- maximum_fire_power: 最大威力 (完成後の ojama 数 = 飽和連鎖量)
- mid_game_response_capacity: 中盤応答能力 (催促打ち合いの余裕)
- harassment_readiness: 催促可能性 (即催促打てる小連鎖の存在)
- ojama_defense_capacity: お邪魔体制 (受けて掘れる量)

**形分類 (form_gtr 等) は補助的役割に格下げ**:
- 既存 form_gtr / form_llr / form_staircase / form_zabuton は「形がそろっていれば、機能達成の可能性が高い」程度の proxy
- 機能指標が直接測れるなら形指標は重複・冗長
- 学習で「形指標が機能指標と相関高ければ削除候補、独立価値あれば保持」を判定

### 段階方針
- Go/no-go ゲート方式 (各 phase 終了時に効果測定 → 続行 or 軌道修正)
- 無駄実装回避
- **基本アーキテクチャ: End-to-end CNN + 補助 indicator (確定)**
  - メイン軸: CNN が直接「盤面 → 勝率」を学ぶ (人間想定超え狙い)
  - 補助軸: 45 indicator は auxiliary supervision target (multi-task loss)
  - オーバーレイ表示用に補助指標を予測ヘッドから取り出し
  - AlphaGo/AlphaZero 系の確立アーキテクチャ
- **Quick mode**: H1 以降は 12 動画 subset で高速イテレーション、最終 Phase L で本番化

### アーキテクチャ図
```
入力: 盤面 (時系列 5-10 frame) + ProbabilisticBoard 確率分布
   ↓
Backbone: 3D CNN または 2D Conv + LSTM
   ↓
   ┌─ Primary Head: 勝率予測 (-100 〜 +100)
   ├─ Auxiliary Head 1: 45 indicator 予測 (回帰)
   ├─ Auxiliary Head 2: ChainSimulator 結果予測
   └─ Auxiliary Head 3: time_phase 推定
   ↓
Multi-task Loss = α·勝率 + β·indicator + γ·simulator + δ·phase
```

### Phase H1-H3 の役割再定義 (end-to-end ベース下)
- H1: 機能能力指標 + 戦況指標 = **auxiliary supervision の target 設計**
- H2: 時系列展開 = auxiliary supervision の時系列 target
- H3: 弱指標削除 = 不要な auxiliary loss を削減
- 旧 2-tower (ハンドクラフト + CNN late fusion) は廃止

### Quick mode 動画選定 (12 動画、確定)
**v01, v04, v07, v12, v20, v22, v28, v40, v51, v57, v70, v89**
- 大型 2 (v04, v12): CNN heavy data
- 中型 6 (v01, v07, v22, v40, v57, v70): 代表構成 (大会別含む)
- 小型 3 (v20, v28, v51): fast iteration
- 多様性 1 (v89): 別 UI
- Phase H1 のみ全 66 動画 (現在進行中、Phase L のベースライン)
- Phase H2-H4 は 12 動画 quick mode (regen ~2h)
- Phase L で 66+追加動画で本番化

### Quick mode の go/no-go 基準 (信頼性確保)
動画数少のため判定は厳しめ:
- LR Phase LOOV avg: **+0.04 以上** (旧 +0.03 から強化)
- GBM video holdout: **+0.04 以上**
- 両方達成で go、片方のみで再評価検討

---

## 階層構造

```
階層 1 (認識): フレーム → 盤面 grid + ProbabilisticBoard
階層 2 (評価): grid + 履歴 → 指標群 → 勝率
階層 3 (出力): 勝率 + 重要指標 → リアルタイムオーバーレイ
```

各階層は独立 phase 群で進化、最終的に統合。

---

## 段階別ロードマップ (合計 5-7 ヶ月)

### 出発点 (現在 2026-05-08)

### Phase J0: 既存資産で評価動画作成 + β版動作確認 (★ 第一目標、1-3 日)

**目的**: まず動かしてみる、評価動画作成、改善ポイント特定

#### 実装
- 既存 `src/cli.py compose` で 1-2 短編動画にオーバーレイ合成
- 現状の Phase E 学習結果 (LR 0.659 旧 / GBM 未統合) を使用
- 既存 `OverlayRenderer` のレンダリング結果確認
- `VideoCompositor` で音声 mux 込みの最終 mp4 生成

#### Deliverables
- 評価動画 1-2 本 (`data/evaluation_videos/v0X_overlay.mp4`)
- レビュー結果: オーバーレイの見栄え、判定精度、誤判定パターン
- UI/UX 改善ポイントリスト

#### 完了条件
- 動画再生で破綻なし (オーバーレイが正しく表示される、音声も保持)
- 主観で「使えそう / 改善要」のフィードバック

#### 次の判断
- 評価動画見て: 精度十分なら Phase J 直行 (オーバーレイ完成へ)
- 改善が要るなら Phase H1 から精度向上へ


- LR Phase LOOV avg **0.652**
- GBM video holdout **0.679**
- 25 indicator + 4 確率版 override
- W-α HiddenRowInferrer pipeline 統合済
- 認識: per-cell 99.5%、STABLE state machine 確定盤面のみ
- BCD CSV: 10,945 行 / 33 列 / 66 動画

### Phase H1: 機能能力指標 7 個 + 戦況指標 8 個 + 既存修正 (1-2 週間)

**目的**: 上級者戦略概念を観測指標化 (機能ベース優先)、学習で重要度判定

#### 一次軸: 機能能力指標 7 個 (★ 形を超える本質指標)

形 (GTR / サブマリン / 階段) でなく、**「達成された機能」**を直接測る:

| 指標 | 観測内容 | 計算方針 |
|---|---|---|
| **ready_chain_count** | 現時点で即発火可能な連鎖数 | ChainSimulator(board) の chain_count |
| **ignition_distance** | 何手で発火可能になるか | 1-step / 2-step 探索で発火寸前を判定 (既存 required_puyo_to_fire 拡張) |
| **current_fire_power** | 即発火した場合の ojama 数 | chain_result.score / OJAMA_DIVISOR |
| **maximum_fire_power** | 完成後の最大 ojama 数 (飽和連鎖量) | 現盤面 puyo の最適配置推定 (heuristic) |
| **mid_game_response_capacity** | 中盤応答能力 (催促打ち合いの余裕) | (sub_chain_quality + maximum_fire_power 残量) / 1催促消費 |
| **harassment_readiness** | 即催促打てる小連鎖 (2-4連鎖) の存在 | sub_chain_quality 拡張、即発火可能小連鎖の威力 |
| **ojama_defense_capacity** | お邪魔体制 (受けて掘れる量) | 仮想 ojama 10-30 落下後、本線回復可能性 |

これらは形分類 (GTR/サブマリン) を超えた「機能の到達度」を直接測定。
形が変わっても、これら 7 指標が同じなら勝率は同等のはず。

#### 二次軸: 戦況・タイミング指標 8 個

機能達成度を文脈化する指標 (時間軸・対戦相互作用):

| 指標 | 観測内容 |
|---|---|
| self_chain_duration_frames | 自連鎖 frame 数 (落下マス × クイック反映) |
| opp_chain_duration_frames | 相手連鎖 frame 数 |
| chain_duration_advantage | 応答可能 puyo 数差 (相手連鎖中に自分が打てる手) |
| harass_event_count_30s | 直近 30s の催促回数 (自/相手) |
| early_aggression_score | +30s での攻撃量 (連鎖発火 + ojama 送信) |
| counter_ignition_signal | ojama 受け直後の本線発火パターン検出 |
| post_all_clear_state | 序盤全消し検出 + ボーナス使用判定 |
| upper_board_density | 上部 (10+ 段) puyo 数 (上級者 advantage 観測) |

#### 三次軸: 形分類指標 (補助、既存 + 1 個追加)

形は機能達成の手段。学習で「機能指標と独立価値あるか」を判定:

| 指標 | 状態 |
|---|---|
| form_gtr (B-1 既存) | 補助、機能指標と相関判定 |
| form_llr (B-1 既存) | 補助 |
| form_staircase (B-1 既存) | 補助 |
| form_zabuton (B-1 既存) | 補助 |
| **gtr_orientation** (新) | 先折り (0) / 後折り (1) / 自由形 (2) |

#### 既存指標修正 (4 個)

| 既存 | 修正内容 |
|---|---|
| main_chain_maturity | chain_power_full と並列保持 (連鎖数 vs 真威力で意味分離) |
| harassment_resistance | defensive_option_quality (受ける選択肢) を追加 |
| key_flexibility | キーぷよ外し本線戦略視点で再設計 |
| death_risk | upper_board_density との合成、純粋な窒息リスクのみに |

#### 設計指針
- 全指標 `opponent_board` 引数 native (相互作用統合)
- API は時系列対応可能設計 (詳細値を `detail` dict に保持、H2 で活用)
- stateless 実装 (state-holding tracker は H2 で外部 wrapper)

#### 学習側の準備
- 全指標の Permutation Importance + SHAP
- Phase Aware で序盤・中盤・終盤の重要度差を可視化
- ablation 自動化 (各指標を消したら何 pt 落ちるか)

#### Deliverables
- `src/indicators.py`: 16 個追加 (機能 7 + 戦況 8 + gtr_orientation 1) + 4 個修正
- `IndicatorSet`: 16 fields 追加
- `EXTRA_INDICATOR_NAMES`: 29 → 45 個
- `tests/test_indicators_capability.py`: 機能指標 30 テスト
- `tests/test_indicators_situational.py`: 戦況指標 25 テスト
- BCDE regen (66 videos) — wall ~10-12h (指標増 + 計算量増)
- 学習 + ダッシュボード `phase_h1_dashboard.md`
- **重要度 ranking 出力**: 機能 vs 戦況 vs 形のどれが効くかの分析

#### Go/no-go ゲート
- LR Phase LOOV avg: **+0.03 以上** (0.652 → 0.685+)
- GBM video holdout: **+0.02 以上** (0.679 → 0.700+)
- 達成不能なら: 個別指標 ablation で原因特定、設計修正

#### 期待効果: GBM **0.679 → 0.72-0.74**

---

### Phase H2: 時系列展開 + データ pipeline 刷新 (2-3 週間)

**目的**: 静的スナップショット → 動的時系列、モメンタム/リズム捕捉

#### データ pipeline 刷新
- 既存: 1 試合 5 行 (5 phase × 1 snapshot)
- 新: 1 試合 30-100 行 (各 STABLE frame で record)
- 各行: 42 静的指標 + 履歴メタデータ

#### 時系列展開 (全 42 指標を 6 軸で展開)
1. 静的値
2. Δ (前 STABLE 比)
3. 加速 (Δ の Δ)
4. 履歴 max (直近 30s)
5. 履歴 min (直近 30s)
6. 平均 (直近 30s)

= **42 × 6 = 252 features**

#### Interaction features (8-12 個明示追加)
- self_main × opp_threat
- self_chain_power × opp_chain_power
- self_saturation × opp_main_chain
- self_harass_density × opp_defense
- 等

#### モデル革新 (一部)
- LightGBM + Mixed-effects 風 (動画別 random intercept)
- video-grouped CV 厳密化

#### Deliverables
- `scripts/phase_h2_collect_indicator_dataset.py` (新パイプライン)
- 新 CSV: `match_features_phase_h2.csv` (10万行級)
- `src/timeseries_indicator_wrapper.py`
- 学習 + ダッシュ `phase_h2_dashboard.md`

#### Go/no-go ゲート: **+0.04 以上**

#### 期待効果: **0.74 → 0.78-0.80**

---

### Phase H3: 弱指標削除 + 評価フレーム強化 (1 週間)

**目的**: 動画間ばらつき (LOOV std 0.07-0.09) 補正、弱指標除去

#### 既存指標再評価
- 全 252 features + interaction の真重要度 (SHAP / Permutation)
- 弱指標 (rank 100+ で常時 < 0.005) 削除候補
- VIF 高い multicollinear pair 圧縮

#### モデル革新 (本格)
- LightGBM + Mixed-effects model (statsmodels.MixedLM)
- video-grouped CV with player matching
- ensemble: GBM + RF + MLP + Mixed-effects

#### 補助タスク
- player クラスタリング (k-means in indicator space)
- 試合中 Win Probability 動的推定 (S-D ライト)

#### Deliverables
- `scripts/phase_h3_ablation.py`
- `data/verify/phase_h3_dashboard.md`
- 削除指標リスト + 残存指標一覧

#### Go/no-go ゲート: **+0.02 以上**

#### 期待効果: **0.80 → 0.82-0.84**

---

### Phase H4.1: C-1 単純 CNN embedding (3 週間)

**目的**: ハンドクラフトを超えるパターンを CNN が発見できるか検証

#### 実装
- 入力: 6×13 整数配列 (色コード)
- アーキ: 2D Conv 3 層 → Flatten → 32 次元 embedding
- 統合: ハンドクラフト 252 features + CNN 32 = 284 features → MLP
- **2-tower late fusion model** (ハンドクラフト tower + CNN tower)

#### 説明可能性維持
- ハンドクラフト tower の SHAP は引き続き使える
- CNN tower は Grad-CAM 可視化
- どちらが効いてるかの貢献度測定

#### Go/no-go ゲート: **+0.02 以上**

#### 期待効果: **0.84 → 0.86-0.88**

---

### Phase H4.2: C-4 完全版 CNN (4-7 週間)

**目的**: 時系列 + 確率分布 + シミ結果を統合した最強 CNN

#### 実装
- 入力: 5-10 frame 時系列 stack
- 各 frame: 6×13×7 channel (色 7 種 × 確率分布) + ChainSimulator 結果 (発火位置、連鎖順序) auxiliary
- アーキ: 3D Conv または 2D Conv + LSTM (multi-input CNN)
- リソース: RTX 4060 Laptop 8GB ぎりぎり、batch_size 32 → 16、mixed precision

#### Go/no-go ゲート: **+0.03 以上**

#### 期待効果: **0.88 → 0.91**

---

### Phase H4.3: 事前学習 + 蒸留 (5-10 週間)

**目的**: 人間想定を超える表現学習

#### 実装
- 大量未ラベル動画 (66 → 数百本) で BYOL/SimCLR 事前学習
- ハンドクラフト指標を CNN に蒸留 (auxiliary loss)
- 学習: 24-48h × 数回 iteration

#### 期待効果: **0.91 → 0.93-0.95+**

---

### Phase I: 認識精度 99.99% 達成 (3-4 週間、Phase H と並列)

**目的**: 上級者対戦の認識ノイズ除去 (学習データ品質向上)

#### 実装
- W-α prob_board の真の効果検証 (現状統合済だが効果未測定)
- Online HSV calibrator (memory `realtime_hsv` 段階 2)
  - 試合中、CNN 確信度 ≥0.95 + HSV 一致サンプルを EMA で蓄積
  - 1-2 試合ウォームアップで動画別 HSV ranges へ自動切替
- 試合開始 BG キャリブ自動化
- 段階別ターゲット: 99.5% → 99.7% → 99.9% → 99.99%

#### Phase H と並行実装可能
- H1-H4 は Phase E パイプライン (静止画像 + STABLE) ベース
- 認識改善は別ファイル (recognition pipeline、HSV calibrator)
- 衝突なし

---

### Phase J: リアルタイムオーバーレイ統合 (2-3 週間、最終目標)

**目的**: 最終ローンチ、配信オーバーレイツール完成

#### 実装
1. 既存 stream_overlay.py の OBS 統合検証 (現状 memory「未検証」)
2. WebSocket 経由でリアルタイム指標配信
3. フロント UI (HTML/JS):
   - 勝率バー (左右、-100 〜 +100)
   - Top 5 重要指標表示 (動的)
   - 戦況フェーズ判定 (序盤/中盤/終盤)
   - 警告通知 (窒息間近、カウンター成功 等)
4. 60fps 入力 → 5-10fps eval pipeline 最適化
5. Latency 200ms 以内目標
6. ChainSimulator キャッシュ + 軽量モデル併用

#### 依存
- Phase H 完成後 (高精度評価)
- 認識 99.7% 以上が望ましい (誤判定が画面に出ると致命的)

---

### Phase K: 配信実証 (1-2 週間)

**目的**: 実戦投入、フィードバック収集、調整

#### 実装
- 大会配信での試験運用
- 視聴者・解説者からのフィードバック収集
- 説明文 / UI 調整
- パフォーマンス計測

---

## マイルストーンとタイムライン

| 時期 (今から) | Phase | 期待精度 | 機能完成度 |
|---|---|---|---|
| 0 (現在) | BCD | 0.679 | 学習基盤 |
| 2 週間後 | H1 | 0.71-0.74 | 上級者戦略指標 |
| 5 週間後 | H2 | 0.76-0.80 | 時系列対応 |
| 6 週間後 | H3 | 0.80-0.83 | モデル革新 |
| 9 週間後 | H4.1 + I 並行 | 0.82-0.85 / 99.7% | CNN 単純 + 認識改善 |
| 13-16 週間後 | H4.2 | 0.85-0.90 | CNN 完全版 |
| 18-22 週間後 | H4.3 + I 完成 | 0.90-0.95+ / 99.99% | 学習完成 + 認識完成 |
| 22-25 週間後 | J | - | オーバーレイ統合 |
| 25-28 週間後 | K | - | 配信実証 |

**実用ローンチ最短**: H3 完了 + I 80% + J = **約 4 ヶ月後** (精度 0.83、認識 99.7%)
**最大規模完成**: 全 Phase 完成 = **約 7 ヶ月後** (精度 0.93+、認識 99.99%)

## リスク & mitigations

| リスク | 影響 | 対策 |
|---|---|---|
| H1 で +0.03 出ない | 戦略指標設計失敗 | ablation で個別切り分け、設計修正 |
| H2 計算量爆発 | regen 不能 | サンプリング (1試合 30 行 → 15 行)、並列化 |
| H4 GPU メモリ不足 | C-4 完全版実装困難 | mixed precision / batch 半減 / cloud GPU |
| Mixed-effects 収束しない | M-C 効果不明 | LightGBM native categorical で代替 |
| 認識 99.99% 達成困難 | overlay 誤判定 | 段階別ターゲット (99.7% でも実用) |
| 動画数不足 (CNN 事前学習) | 表現学習限定 | yt-dlp で大会動画追加 |
| Overlay latency 200ms 超 | UX 損 | C++/Rust 化、量子化、キャッシュ |
| 上級者試合データ偏り | プレイヤー識別困難 | 大会別・プレイヤー別 holdout |

## 確定事項

- ✅ 上級者対戦の有利不利判定スコープ
- ✅ 「観測軸を提供 → 学習で重要度発見」設計思想
- ✅ Phase 別 go/no-go ゲート方式
- ✅ ハンドクラフト → CNN embedding 段階移行
- ✅ 説明可能性維持 (ハンドクラフト tower)
- ✅ リアルタイムオーバーレイが最終目標
- ✅ 認識 99.99% 目標 (Phase I で並列追求)

## 関連ファイル

- 現状ダッシュボード: `data/verify/phase_e_dashboard_bcd.md`
- 学習 audit: `data/verify/learning_impact_audit.md`
- ハンドオフ: `docs/SESSION_HANDOFF_2026-05-06.md`、`memory/project_handoff_2026-05-07_bc.md`
- ロードマップ (本): `docs/INDICATOR_ROADMAP.md`

---

**END OF ROADMAP**
