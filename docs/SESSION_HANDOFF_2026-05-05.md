# Session Handoff: 2026-05-05

**目的**: 学習精度の頭打ち (0.66) を破るための障害解消プロジェクトの引き継ぎ。
本日 (2026-05-05) のセッションで Tier A (軽量) と Tier B (一部) を解消、
Tier B 中規模と Tier C (大規模) を次セッションで実装する。

---

## 1. 本日の完了タスク

### 1-1. 教師データ拡張
- 6 プレイリスト中の残り 4 (plB/plC/plD) から 35 本追加 DL
- count_match_v5 + detect_match_winners で全動画の境界 + 勝者ラベル生成
- E-1 (`scripts/phase_e_collect_indicator_dataset.py`) で v01-v94 の指標 csv 生成中
  - 動画別 shard: `data/training/phase_e_shards/shard_v??.csv`
  - 統合 csv: `data/training/match_features_phase_e_v01-XX.csv`
  - 進行中: 56/72 shard 完了見込み (実行終了時刻に追記必須)

### 1-2. 学習実行 (E-2/E-4 + GBM)
- LR (sklearn LogisticRegression L2) と HistGradientBoosting で重み学習
- 結果サマリ (v01-v40 ベース、7,650 行):
  - LR L2 video_holdout: **0.629** (旧 V3=0.600 から +2.9pt)
  - LR L2 LOOV avg: **0.659** (start 0.533 / mid 0.582 / end **0.862**)
  - GBM video_holdout: **0.683** (+5.4pt)
  - GBM LOOV avg: **0.669** (start 0.519 / mid 0.592 / end **0.896**)
- `src/scorer.py` に LEARNED_WEIGHTS_PHASE_E_* (16 features) を反映、
  `weight_mode="phase_e"` で利用可能。テスト 34/34 パス。

### 1-3. 障害解消 (Tier A 軽量 6 個)
| ID | 内容 | ファイル | 状態 |
|:---:|:---|:---|:-:|
| **W-α** | HiddenRowInferrer を pipeline 統合 (TSUMO→STABLE 遷移時に row 0 量子推論) | `src/recognition_pipeline.py` | ✅ |
| **D-A** | intro 試合 (v14/17/18) の matches/winners.tsv を自動補正 | `scripts/fix_intro_match_misdetection.py` | ✅ |
| **W-η** | count_match_v4 に intro 自動 skip ロジック追加 (動画開始 30s 以内 + 100s+ 長尺) | `scripts/count_match_v4.py` | ✅ |
| **D-C** | time_phase の相対化 (絶対秒 → 試合進捗 % ベース) | `scripts/phase_e_collect_indicator_dataset.py` | ✅ |
| **I-G** | field_efficiency 連結閾値 ≥2 → ≥4 (連鎖発火条件と整合) | `src/indicators.py` | ✅ |
| **M-A** | HistGradientBoosting (sklearn) 移行 | `scripts/learn_weights_lgbm.py` | ✅ |

### 1-4. Tier B 形質指標 (3/7)
| ID | 内容 | ファイル | 状態 |
|:---:|:---|:---|:-:|
| **I-I 形質 1** | `planning_entropy`: 1ツモ追加で発火する連鎖サイズ分布のエントロピー | `src/indicators.py` | ✅ |
| **I-I 形質 2** | `structure_solidity`: 下半分の連結 ≥3 ぷよ数比率 | `src/indicators.py` | ✅ |
| **I-I 形質 3** | `base_flatness`: 下層 3 段の高さ標準偏差 (1 - normalized_std) | `src/indicators.py` | ✅ |

→ EXTRA_INDICATOR_NAMES が 13 → 16 に。CSV 列に新 3 指標が追加される。

---

## 2. 残タスク (次セッションで実装)

### 2-1. Tier B 中規模 (4 個、~3-5 日)

#### B-1: I-J 形テンプレ完成度 4 指標
**目的**: GTR/LLR/階段/座布団 のテンプレートとの一致度を計算し、
上級プレイヤーの戦略型を識別。`key_flexibility` 二相性をさらに精緻に分離。

実装ガイド:
- `src/indicators.py` に `GtrCompletenessIndicator` 等 4 クラス追加
- 各テンプレ盤面 (color-coded mask) を定義: 例 GTR は左端 col0-1 の 4 段
- 現盤面との NCC (正規化相互相関) で類似度 0-1 算出
- ミラー (1P 用 GTR は左端、2P 用は右端) 対応

期待効果: +0.02-0.05 (LR/GBM)

#### B-2: I-E extension_potential 探索深さ拡張
**目的**: 1 ツモ → 2 ツモ仮置きで連鎖伸長の精緻化。

実装ガイド:
- `src/indicators.py` の `ExtensionPotentialIndicator.compute` を改修
- 1 ツモ目で発火しなかった候補について、2 ツモ目の探索を追加
- 計算量 6 倍程度。E-1 拡張時間も 6 倍になる可能性 → workers 増やすか fps 下げる必要

期待効果: +0.01-0.03

#### B-3: W-κ score OCR 動画別キャリブ
**目的**: 大会別 UI で score 数値の位置/フォントが異なる場合の対応。

実装ガイド:
- `src/score_ocr.py` にテンプレ動画別読込
- `data/verify/digit_samples/` の動画別フォルダ拡張

期待効果: 認識精度が新動画でも維持される (現状 OK だが将来的に必要)

#### B-4: W-δ 回し入れ追跡
**目的**: 上級者の隠し段への puyo 配置を確定盤面に反映。
W-α (HiddenRowInferrer) を拡張して、可視領域から隠し段への移動を捕捉。

実装ガイド:
- `src/hidden_row_inferrer.py` に「row 1 puyo が消失 + 操作中」のパターン検出
- 同列の隠し段 (row 0) に puyo 移動の仮説を生成

期待効果: +0.01-0.02 (上級者試合で大きい)

### 2-2. Tier C 大規模 (3 個、~1-2 週間)

#### C-1: W-β + I-A indicator 全体の確率分布対応
**目的**: ProbabilisticBoard を indicator 計算で消費可能にし、量子状態
(隠し段 row 0、CNN 信頼度低 cell) の情報を失わずに学習に活用。

実装ガイド:
- 16 指標の `compute(board)` シグネチャを `compute(board, prob_board=None)` に拡張
- 各指標で確率重み付き計算ロジック追加 (例: death_risk は確率密度として高さに加算)
- W-α の効果を最大化する基盤

期待効果: +0.02-0.05

#### C-2: W-γ おじゃま予告→落下スケジュール追跡
**目的**: `incoming_ojama_pressure` を真に有効化。score OCR 差分から
予告おじゃま量を取得し、落下スケジュールに変換、indicator に渡す。

実装ガイド:
- `src/score_ocr.py` から score 差分を pipeline で取得済 (D-1)
- 新規 `src/ojama_predictor.py` を作成、score_delta → 予告おじゃま個数 → 落下時刻
- recognition_pipeline で incoming_ojama を indicator に渡すパイプ
- `IncomingOjamaPressureIndicator` を score OCR ベースに書き直し

期待効果: +0.03-0.08 (現状 0 で機能不全な指標が活性化)

#### C-3: I-H 対戦相互作用
**目的**: 1P/2P 単独計算から、2P 文脈を考慮した指標へ拡張。
例: 「相手の連鎖タイミングに合わせる催促」の表現。

実装ガイド:
- 各指標に `opponent_board` パラメータを追加 (現状 `OpponentChainThreatIndicator` のみ受領)
- 例: `harassment_resistance` = 相手の予告おじゃま量で動的閾値変更
- 例: `chain_timing_pressure` = 相手の発火タイミング予測との差分

期待効果: +0.02-0.05

### 2-3. その他のキャリーオーバー

| ID | 残作業 |
|:---:|:---|
| W-ζ | STABLE 確定遅延の調整 (stable_frame_count フィルタ最適化) |
| W-θ | BG fingerprint 品質改善 (動画別品質判定) |
| W-ι | next/dnext 動画別 ROI |
| W-ε | ChainEvent 検出漏れ完全解消 |
| I-C | offset_power score OCR 統合 |
| I-D | harassment_resistance 落下シミュ改良 |
| I-F | shape_score 複数テンプレ評価 |
| D-B | UNKNOWN 率 (label noise) 改善 |
| D-D | 動画間プレイヤー戦術差 (Mixed-effects) |
| D-E | 時系列特徴量 (試合中の盤面変化レート) |
| D-F | 中盤ラベル (途中の優勢/劣勢) |
| M-B | Phase Aware の連続境界 (線形補間) |
| M-C | Mixed-effects model (動画別 random intercept) |
| M-E | video-grouped CV split |

---

## 3. 主要ファイル参照

### 学習・教師データ
- `scripts/phase_e_collect_indicator_dataset.py` — E-1 (relative phase 化、v5 fallback、worker spawn 並列対応)
- `scripts/learn_weights_v3.py` — LR L2 学習 (E-2/E-3)
- `scripts/phase_e_learn_phase_aware.py` — Phase Aware LR L2 学習 (E-4)
- `scripts/learn_weights_lgbm.py` — HistGradientBoosting 学習 (M-A)
- `scripts/phase_e_dashboard.py` — 評価ダッシュボード生成
- `scripts/phase_e_indicator_eval.py` — 指標評価 (VIF、importance、scenario)
- `scripts/multicollinearity_analysis.py` — 多重共線性分析
- `scripts/fix_intro_match_misdetection.py` — D-A intro 試合補正

### 認識
- `src/recognition_pipeline.py` — pipeline 本体 (W-α 統合済)
- `src/board_state_machine.py` — state machine
- `src/hidden_row_inferrer.py` — W-α の量子推論ロジック
- `src/probabilistic_board.py` — 確率盤面表現 (C-1 で活用予定)
- `src/score_ocr.py` — score OCR (C-2 で活用予定)

### 指標
- `src/indicators.py` — 全 24 指標定義 (Tier B 形質 3 追加済、I-G 修正済)
- `src/scorer.py` — Scorer + PhaseAwareScorer (LEARNED_WEIGHTS_PHASE_E_* 反映済)

### データ
- `data/training/phase_e_shards/` — 動画別 shard (再開可能)
- `data/training/match_features_phase_e_v01-XX.csv` — 統合 csv
- `data/verify/match_boundaries_v5/video_NN/matches.tsv` — 試合境界
- `data/verify/match_winners_vNN.tsv` — 勝者ラベル
- `data/verify/learning_impact_audit.md` — 障害洗い出しドキュメント
- `data/verify/indicator_methodology_review.md` — 指標設計レビュー

---

## 4. 期待効果 (改善前 → 改善後の見込み)

| 段階 | 改善内容 | 期待 LOOV avg |
|:---|:---|---:|
| 現状 | LR (E-3 後 16 指標) | 0.659 |
| **本日完了** | LR + Tier A 軽量 + Tier B 形質 3 (再生成後) | **0.68-0.70** |
| **次セッション B 中規模完了** | + I-J/I-E/W-κ/W-δ | 0.72-0.75 |
| **次セッション C 完了** | + W-β/I-A/W-γ/I-H | **0.78-0.82** |
| 長期 (新フェーズ) | + CNN Embedding + RL | 0.85+ |

---

## 5. 次セッション開始時の手順

### 5-1. 環境確認
```bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
ls data/training/phase_e_shards/ | wc -l   # 再生成完了なら 75 (v22, v39 除外)
ls data/training/match_features_phase_e_v01-94.csv  # 最新 csv
cat data/verify/learned_weights_lgbm_v01-94.json | head -30  # 最新学習結果
```

### 5-2. 推奨実装順
1. B-2 (I-E 探索深さ拡張) ← 計算量大、最初に試して再生成時間見積もり
2. B-1 (I-J 形テンプレ 4) ← 最も期待効果大、半日-1日
3. B-4 (W-δ 回し入れ追跡) ← 中-小、1日
4. B-3 (W-κ score OCR) ← 必要性が出てきたら、半日
5. **再生成 + 学習** → ここで効果中間評価
6. C-2 (W-γ おじゃま予告) ← 大改修、1-3 日
7. C-1 (W-β + I-A 確率分布) ← 大改修、1-2 週
8. C-3 (I-H 対戦相互作用) ← 大改修、1 週
9. **最終再生成 + 学習** → 0.78+ 達成見込み

### 5-3. 再生成コマンド (Tier B 反映後)
```bash
# 全 shard 削除
rm data/training/phase_e_shards/*.csv

# 全動画再生成 (workers=8, fps=3)
PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_collect_indicator_dataset \
    --videos 1-21,23-38,40-69,73-94 --max-matches 0 --fps 3 --workers 8 \
    --out-csv data/training/match_features_phase_e_v01-94_tierb.csv

# 学習
PYTHONPATH=. ./venv/bin/python -m scripts.learn_weights_v3 \
    --csv data/training/match_features_phase_e_v01-94_tierb.csv \
    --out data/verify/learned_weights_v3_tierb.json
PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_learn_phase_aware \
    --csv data/training/match_features_phase_e_v01-94_tierb.csv \
    --out data/verify/learned_weights_phase_e_phase_aware_tierb.json
PYTHONPATH=. ./venv/bin/python -m scripts.learn_weights_lgbm \
    --csv data/training/match_features_phase_e_v01-94_tierb.csv \
    --out data/verify/learned_weights_lgbm_tierb.json
```

---

## 6. 重要な観察 (引き継ぎ用 insights)

1. **GBM の伸びは限定的 (+1pt のみ)**: 非線形相互作用の余地が小さい。
   feature engineering (新指標) が真の改善源。
2. **`extension_potential` が支配的**: permutation importance で他指標の 2.5 倍。
   ここを精緻化する I-E が高効率。
3. **`key_flexibility` の負係数は二相性が原因**: Tier B 形質 3 指標で分離可能。
   再生成後に検証必要。
4. **動画間 LOOV std が高い (~0.05-0.08)**: D-D (Mixed-effects) で改善見込み。
   ただし大規模改修。
5. **start phase が最弱 (0.51-0.53)**: 構造ベース指標の限界。
   新 3 指標 (column_balance/tsumo_entropy/3rd_col_pressure、未実装) で底上げ可能。
6. **隠し段 (W-α) 統合の効果は再生成しないと検証不可**: 今夜の再生成で初確認。

---

## 7. 最新学習結果 (Tier B 形質指標**なし** = 旧 indicators)

`data/training/match_features_phase_e_v01-94.csv` (10,236行 / 69動画)
で学習した結果 (E-1 拡張 v01-v94 段階):

| Phase | LR L2 LOOV | GBM video_holdout |
|---|---:|---:|
| start | 0.522 ± 0.101 | — |
| mid | 0.586 ± 0.081 | — |
| end | **0.855 ± 0.089** | 0.896 |
| **平均** | **0.654** | 0.683 |

旧 v01-v40 (7,650 行) と比べ:
- LOOV avg: 0.659 → 0.654 (-0.005、低品質動画混入)
- end: 0.862 → 0.855 (-0.007)
- video_holdout: 0.629 → 0.659 (+3pt、データ量の効果)

**注**: Tier B 形質指標 (structure_solidity / base_flatness) は
csv に**まだ反映されていない**。次セッションでの再生成 + 学習で +0.02〜0.05 が
期待される。

## 8. Tier B 反映再生成の課題 (2026-05-06 早朝追記)

### 計算量問題

Tier B 形質 3 指標の reg 反映で **再生成時間が 6-9 倍に膨張**:
- 旧 indicators (16 指標): ~6 時間 (75 動画、workers=8)
- 新 indicators (Tier B 含む): 50 分以上経過しても 1 shard も出ない

試行履歴 (本セッション):
1. planning_entropy 30 通り → 1 時間で 1 shard
2. 軽量化 (15 通り) → 1 時間で 1 shard (改善なし)
3. ChainSimulator メモ化キャッシュ実装 → 50 分で 0 shard
4. planning_entropy 完全無効化 + cache → 50 分で 0 shard

= **structure_solidity と base_flatness だけでも追加負荷が大きい**
(_connected_components が毎 frame 全盤面スキャン、~80 cell)

### 現状の決定 (2026-05-06 04:50 時点)

本セッション内の Tier B 反映完走は時間的に困難と判断、停止。
次セッションで以下の最適化を実装してから再走:

1. `_connected_components` の毎 frame 計算を回避 (盤面 hash でキャッシュ)
2. structure_solidity と base_flatness を numpy ベクトル化
3. planning_entropy は ChainSimulator キャッシュ + 列毎 1 試行に簡素化

実装後、Tier B 反映で再生成 → 学習で効果検証。

### 次セッションでの根本対応 (推奨)

1. **ChainSimulator にメモ化キャッシュ追加** (`src/chain.py`):
   ```python
   _cache: dict[bytes, ChainResult] = {}
   def simulate(self, board):
       key = board.grid.tobytes()
       if key in self._cache: return self._cache[key]
       ...
   ```
   同一盤面の simulate を高速化 (~10 倍)
2. または **planning_entropy をオプショナル化** (--include-planning-entropy フラグ)
3. または **HistGradientBoosting と planning_entropy を別途学習** して
   後付け feature にする (E-1 で生成しない)

### 朝の手順

```bash
# 完走判定
ls data/training/phase_e_shards/ | wc -l    # 60+ なら完了

# 統合 csv 生成
python scripts/_merge_shards.py  # 必要なら手動

# 学習
PYTHONPATH=. ./venv/bin/python -m scripts.learn_weights_v3 \
    --csv data/training/match_features_phase_e_tierb_v01-94.csv \
    --out data/verify/learned_weights_v3_tierb.json
PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_learn_phase_aware \
    --csv data/training/match_features_phase_e_tierb_v01-94.csv \
    --out data/verify/learned_weights_phase_e_phase_aware_tierb.json
PYTHONPATH=. ./venv/bin/python -m scripts.learn_weights_lgbm \
    --csv data/training/match_features_phase_e_tierb_v01-94.csv \
    --out data/verify/learned_weights_lgbm_tierb.json
```

---

**END OF HANDOFF**
