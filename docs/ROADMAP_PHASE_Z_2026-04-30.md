# Phase Z ロードマップ — 全動画 99.5% 認識率達成 (2026-04-30 策定)

## 1. 目標

**全動画で画面内全情報を cell 単位 99.5% 以上で抽出する**ことを必須条件とする。
これを満たすまで RL / オーバーレイ / 別大会動画の追加には進まない。

## 2. 評価メトリック (公式)

| 項目 | 定義 |
|---|---|
| 単位 | cell 単位 (フィールド 78 cells / frame) |
| サンプリング | 0.1s ごとに 5 frame 取得 → 0.5s ごとに majority vote で確定 |
| 判定 | 0.5s ごとの「確定 cell」が GT と一致する比率 ≥ 99.5% |
| 対象動画 | v01..v19 全 19 本 |
| ベースライン動画 | `full_review_v18_m03_w15fixed.mp4` 30 秒以降を主観察対象 |

sparse 評価ハーネス (`phase_w_eval_cross_video_full.tsv` 等) は補助、main metric ではない。
連続 frame 動画ベースの誤検出率を一次指標とする。

## 3. 検出対象 (画面内すべて)

| ID | 対象 | 現状 | Phase Z での位置付け |
|---|---|---|---|
| F1 | フィールド 6×13 + 隠し段 | CNN v16 + 補正多段、動画 ~70% | **最優先** |
| F2 | Next pair (1P/2P) | CNN+HSV+Centroid 多数決 + StableNext | 強化 (PairLandingCheck と統合) |
| F3 | DNext pair (1P/2P) | 同上、ROI のみ別 | 強化 (Next ↔ DNext shift 整合) |
| F4 | Score (8 桁) | NCC OCR、conf≥0.55 | 取りこぼし削減 (Telop / 連鎖アニメ中) |
| F5 | 予告お邪魔 (pending) | score 差分推論のみ | **視覚予告クロスチェック追加** |
| F6 | 累積スコア (連鎖中計算式) | 未実装 | **新規** |
| F7 | 全消しストック | 未実装 (フラグ管理のみ) | **新規** (視覚 + フラグ) |
| S1 | 試合状態 (中/終了/ポーズ) | MatchEnd + MatchState | 拡張 (リトライ/設定) |
| S2 | テロップ被覆 | TelopDetector | 完全マスク化 |
| S3 | エフェクト (連鎖閃光・爆発) | AnimationFilter のみ | 強化 (motion + UI AND) |

## 4. フェーズ構成

### Phase Z-1: 可視化レビューツール (最優先、1 セッション)

**目的**: ユーザーが任意動画の任意区間を **0.5s 単位で cell ごとに目視レビュー**できる UI を作る。
これがないと Z-2 以降の補正効果が定量化できない。

#### 成果物
- `scripts/phase_z_review_ui.py` — 動画 + 指定区間 (例: v18_m03 30-60s) を読み込み、以下を生成:
  - **0.5s ごとのフレーム**: majority vote 後の確定盤面 + 元画像 + Next/DNext/Score/Pending を 1 枚にレイアウト
  - **誤検出ハイライト**: 物理ルール (PairLandingCheck/Connectivity) で suspicious な cell に枠
  - **比較ペイン**: t-1 / t / t+1 の 3 連続を並べて時系列違和感をすぐ発見できる
- `data/verify/phase_z_review/{video}_{start}_{end}/` 出力ディレクトリ
  - `frames/0030_500.png` (0.5s 単位、ファイル名は ms)
  - `labels.csv` (cell 単位 GT を編集できる csv、char-coded E/R/B/G/Y/P/O)
  - `summary.html` (一覧)
- `scripts/phase_z_apply_review.py` — labels.csv の編集結果を取り込んで accuracy を再計算

#### 半自動 GT 生成
- prev frame + next_pair → ChainSimulator で物理的に許される盤面を予測
- 観測との差分セルのみ "suspicious" としてレビュー対象に
- 予測通りのセルは GT として自動採用 (信頼区間)
- これにより全 cell 人手レビューを 1/10 のコストに圧縮

#### 完了基準
- v18_m03 の 30-60s 区間 (60 frames × 78 cells = 4680 cells) を半自動 GT 化
- 動画別 accuracy (cell 単位、0.5s 多数決) のベースラインが算出される
- ユーザーが追加動画を 1 コマンドで review にかけられる状態

---

### Phase Z-2: ネクスト ↔ フィールド物理推論強化 (2 セッション)

**目的**: ペアの 2 セル同時落下、prev next → cur next の 1 step shift などの**強拘束**を入れる。

#### 新規モジュール
| モジュール | 役割 |
|---|---|
| `src/pre_drop_tracker.py` | NEXT 枠から消えた瞬間 → 落下中フラグ → 着地検出 (motion 停止 + 2 cell 同時出現) |
| `src/pair_drop_physics.py` | 着地は 2 cell 同時、色は prev_next と一致、列は隣接、回転連続的 |
| `src/next_rotation_shift.py` | 着地イベント時 cur_next が prev_dnext と一致するか整合チェック → 不一致なら Next/DNext どちらかを再評価 |
| `src/pair_appearance_enforcer.py` | 1 cell だけ新規出現は **物理不可** → 隣接で検出漏れと判断、補完 |

既存 `pair_landing_check.py` / `enhanced_board_tracker.py` / `next_linked_refiner.py` は残しつつ、より厳密な拘束を追加。

#### 完了基準
- v18_m03 30-60s の Z-1 ベースラインから誤検出率が **半減以上**
- 新モジュール単体テスト全 pass、既存 996 tests 維持

---

### Phase Z-3: 時系列強拘束 (1 セッション)

**目的**: フリッカー除去、落下完了後のみ確定。0.5s 多数決メトリックの精度を直接支える。

#### モジュール強化
| モジュール | 強化内容 |
|---|---|
| `src/temporal_voting_refiner.py` (既存拡張) | window=3 → 5 (= 0.5s) に上げ、majority threshold を導入 (3/5 以上で確定) |
| `src/drop_completion_detector.py` (新規) | phaseCorrelate でフィールド motion を推定、停止 frame のみ "stable" と判定 |
| `src/chain_event_lockdown.py` (新規) | 連鎖発火 (score 急増 + motion) から完了 (score 安定 + motion 停止) まで観測棄却、ChainSimulator 予測採用 |

#### 完了基準
- フリッカー (連続 5 frame で色が振動する cell) の比率が **0.1% 以下**
- chain 発火時の盤面が観測ノイズに振り回されない

---

### Phase Z-4: 誤検出ゼロ化 (1 セッション)

**目的**: EMPTY を puyo と誤認するパターンを徹底排除。

| 対策 | 実装 |
|---|---|
| 完全 UI/Telop マスク | `models/ui_templates/` 拡充 (×印、ヒント、コンボ表示、リプレイ、設定アイコンを 19 動画分) |
| エフェクトマスク | 連鎖閃光・爆発・落下軌跡を motion 急変で UNKNOWN 化 (UI AND) |
| 背景再キャリブ | 試合途中で背景変化検出時に局所更新 (キャラ大幅動作対応) |
| 低彩度強制 EM | cell 中心 S < 厳しい閾値 + V 中域なら強制 EM |

#### 完了基準
- 誤検出 (EMPTY → puyo) の比率が **0.1% 以下**

---

### Phase Z-5: 検出漏れゼロ化 (1 セッション)

**目的**: puyo を EMPTY と誤認するパターンを徹底排除。

| 対策 | 実装 |
|---|---|
| 隠し段 (row 0) OJM 落下中追跡 | 視覚お邪魔予告と照合、Z-6 と並行 |
| ペア落下中 (行間) | Z-2 の PreDropTracker と統合 |
| 特殊演出復帰 | 全消し後の盤面再構築、リトライ後の初期化を MatchEndDetector と連携 |
| 境界 cell | ROI 切り出しで欠ける問題、CELL_SAMPLE_RATIO の動的調整 |

#### 完了基準
- 検出漏れ (puyo → EMPTY) の比率が **0.1% 以下**

---

### Phase Z-6: 画面内全 UI 抽出 (1-2 セッション)

**目的**: 「画面内すべて」のスコープを完成させる。

| 対象 | 実装 |
|---|---|
| 視覚予告お邪魔アイコン (F5) | 上部の予告 (小/大/岩/星/月/王冠) を template + centroid で読み、score 差分とクロスチェック |
| 累積スコア (F6) | 連鎖中の "1340" "+5800" 計算式を ROI 抽出 + OCR、score_ocr の補完として |
| 全消しストック (F7) | 「ALL CLEAR!」表示後の星アイコン視覚検出 + フラグ管理 (`scoring.py` の ALL_CLEAR_BONUS と統合) |
| リトライ/設定/ポーズ | テンプレ NCC マッチ追加、試合中でない区間を MatchStateDetector に統合 |

#### 完了基準
- 5 項目すべての検出が新規モジュールでテスト pass
- ベースライン動画で全項目が確認できる

---

### Phase Z-7: 全動画 99.5% 達成検証 (継続)

- 19 動画全てに Z-1 のレビュー UI を回し、cell 単位 accuracy を計測
- 99.5% 未達の動画には:
  1. user review labels の追加投入
  2. 動画別の弱点特定 (UI 差・解像度・配信者スキン)
  3. CNN 再訓練 (v17, v18, …) または centroid 別個構築
- **達成判定**: 19 動画すべてで cell-level 0.5s 多数決 ≥ 99.5%

---

## 5. ファイル設計

### 新規スクリプト
- `scripts/phase_z_review_ui.py` — 可視化レビュー UI 生成
- `scripts/phase_z_apply_review.py` — labels.csv → accuracy 再計算
- `scripts/phase_z_eval_continuous.py` — 連続 frame 動画ベース評価ハーネス
- `scripts/phase_z_baseline_v18_m03.py` — v18_m03 30-60s ベースライン専用

### 新規 src モジュール (Phase Z-2..Z-6)
- `src/pre_drop_tracker.py`
- `src/pair_drop_physics.py`
- `src/next_rotation_shift.py`
- `src/pair_appearance_enforcer.py`
- `src/drop_completion_detector.py`
- `src/chain_event_lockdown.py`
- `src/visual_ojama_warning.py` (F5 拡張)
- `src/cumulative_score_reader.py` (F6)
- `src/all_clear_stock_detector.py` (F7)

### 出力ディレクトリ
- `data/verify/phase_z_review/{video}_{start}_{end}/`
- `data/verify/phase_z_results/{video}/eval_continuous.tsv`
- `models/ui_templates/phase_z/` (Z-4/Z-6 で追加するテンプレ群)

---

## 6. 進行順序

```
Z-1 (可視化レビュー UI、最優先)
  └─→ ベースライン定量化
      ├─→ Z-2 (Next ↔ Field 物理推論)
      ├─→ Z-3 (時系列強拘束)
      ├─→ Z-4 (誤検出ゼロ化)
      ├─→ Z-5 (検出漏れゼロ化)
      └─→ Z-6 (画面内全 UI)
          └─→ Z-7 (全動画 99.5% 達成検証)
```

Z-2..Z-6 は Z-1 のベースラインが取れた後、効果が大きい順に並列着手可。
Z-7 は Z-2..Z-6 を回しながら継続評価。

## 7. マイルストーン目安

| 日付 | 完了予定 |
|---|---|
| 2026-05-01 | Z-1 完成 (v18_m03 30-60s ベースライン取得) |
| 2026-05-04 | Z-2 完成 (ネクスト連動物理推論) |
| 2026-05-06 | Z-3 完成 (時系列強拘束) |
| 2026-05-08 | Z-4 完成 (誤検出ゼロ化) |
| 2026-05-10 | Z-5 完成 (検出漏れゼロ化) |
| 2026-05-13 | Z-6 完成 (画面内全 UI) |
| 2026-05-20 | Z-7 達成 (全 19 動画で 99.5%) |

短縮される可能性は十分あるが、「99.5% を満たすまでは止めない」方針なので未達なら延長する。

## 8. ユーザールール

- 自律運転 OK
- 「明らかに良い選択肢は聞かずに進める」
- レビュー画像は Windows パスで提示
- **本番統合動画はまだ作らない、認識品質強化を最優先**
- **Phase Z 完了 (99.5% 全動画達成) まで RL / オーバーレイには進まない**
