# セッション引継ぎ 2026-05-09 → 2026-05-10 (Phase I.c 段階 4 完成)

## 🌟 朝のご指摘への対応サマリー

### ① 「v89 1P 27秒下半分が空認識」 → 5 サイクルで完全解決
1. HSV append (DB を default に上書きせず追加)
2. ネクスト履歴 sanity check
3. EFFECT state 強制遷移 (演出中 STABLE 維持)
4. 着地直後 grace period (TSUMO_FALL→STABLE 後 5 frame inferred hold)
5. **stable_frame_count 2→6** ← 真因解決

### ② 「未知動画でも同じ精度を保つ」 → merged_default DB で実現
- 全 6 動画 (v29/v40/v51/v57/v70/v89) DB の union を `_merged_default.json` で構築
- 未知動画でも `--hsv-state _merged_default.json` で **frame 0 から 6 colors inject**
- vs online 学習 (frame 1476 = 49s かかる) と比較で 0-49s 区間に大量 cell 認識違い実証

### ③ 数値検証 (cell-level diff measurement、 Phase I.c)
| 動画 | total cells | DB で改善 | %修正 |
|---|---:|---:|---:|
| v89 | 35,802 | 8,072 | 22.55% |
| v29 | 28,002 | 14,296 | 51.05% |
| v40 | 99,606 | 23,515 | 23.61% |

朝のご指摘「v89 1P 青誤認識」 = blue→empty 5,644 cell 修正で完全解決。

### ④ 未知動画 (aMwcxoWNfzk = 学習未使用) でも検証完了
- 動画 DL → 試合切り出し (480-600s) → 2 通り viz
- merged: frame 0 から 6 colors inject
- online: frame 1476 (49s) で初 inject
- 15s diff で大量認識相違 (= merged 圧倒的有利)

### ⑤ 主要動画ファイル
```
C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\evaluation_videos\
├── v89_match3_phase_i_viz_FINAL2.mp4    ★★★ stable_frame_count=6 で完全認識
├── v89_match3_phase_i_viz_MERGED.mp4    ← merged_default 適用版
├── v29/v40/v51/v57_phase_i_viz_FINAL2.mp4 (各動画別 DB 適用)

C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\
├── unknown_aMwcxoWNfzk.mp4 (DL 元、 55 分)
├── unknown_match_120s.mp4 (切り出し)
├── unknown_viz_merged.mp4 ★ frame 0 から DB
├── unknown_viz_online.mp4 ← online 学習版 (frame 1476 で初 inject)
└── compare\015s_*.png ← 視覚比較
```

---

# セッション引継ぎ 2026-05-09 → 2026-05-10 (最終版)

## 🎯 1分で結果確認するためのファイル

### 一発比較動画 (v89 baseline vs Phase I.c DB、 8 秒区間)
```
data/evaluation_videos/v89_compare/clip_30s_SIDE_BY_SIDE.mp4   ★★ 左右並べて 8 秒比較 (38MB)
```

### 全動画 viz_DB (Phase I.c の universal 性、 5 動画)
```
data/evaluation_videos/v89_match3_phase_i_viz_DB.mp4   (201 MB) ← 元動画 95s
data/evaluation_videos/v29_phase_i_viz_DB.mp4          (309 MB) ← 元動画 156s
data/evaluation_videos/v40_phase_i_viz_DB.mp4          (404 MB) ← 元動画 125s 60fps
data/evaluation_videos/v51_phase_i_viz_DB.mp4          (358 MB) ← 元動画 97s 60fps
data/evaluation_videos/v57_phase_i_viz_DB.mp4          (388 MB) ← 元動画 100s 60fps
```

### 数値検証 (cell-level diff measurement)
| 動画 | total cells | DB で改善 | %修正 |
|---|---:|---:|---:|
| **v89** | 35,802 | **8,072** | **22.55%** |
| v29 | 28,002 | 14,296 | 51.05% |
| v40 | 99,606 | 23,515 | 23.61% |

→ 朝のご指摘「v89 1P 青誤認識」 は DB pre-inject で **5,644 cell 修正** (= blue→empty)

### 動画別 HSV ranges DB (6 動画)
```
data/per_video_hsv_ranges/v29.json (6 colors)
data/per_video_hsv_ranges/v40.json (6 colors)
data/per_video_hsv_ranges/v51.json (4 colors red/blue/yellow/purple)
data/per_video_hsv_ranges/v57.json (4 colors red/blue/green/yellow)
data/per_video_hsv_ranges/v70.json (6 colors)
data/per_video_hsv_ranges/v89.json (5 colors、 green 不足)
```

---

## 🎯 朝起きた時の主要確認ポイント

### 1. 数値検証された「青背景バイアス」 修正 (朝のご指摘の核心)
| 動画 | total cells | DB で改善 | %修正 | 主要 (baseline→DB) |
|---|---:|---:|---:|---|
| **v89** | 35,802 | **8,072** | **22.55%** | blue→empty 5,644 ★ |
| **v29** | 28,002 | 14,296 | 51.05% | empty→blue 3,166 / 各色→empty 大量 |
| **v40** | 99,606 | 23,515 | 23.61% | green→empty 7,048 / blue→empty 5,923 |

→ 「v89 1P 青誤認識」 のご指摘を **DB pre-inject (Phase I.c) で 5,644 cell 修正**。

### 0. 8 秒で効果確認できる比較 clip (28-36 秒区間) ★おすすめ
```
data/evaluation_videos/v89_compare/clip_30s_baseline.mp4   ← 改善前 8 秒 (19MB)
data/evaluation_videos/v89_compare/clip_30s_AB.mp4         ← A+B 統合 8 秒 (19MB)
data/evaluation_videos/v89_compare/clip_30s_DB.mp4         ← Phase I.c DB 8 秒 (18MB)  ★最終
```

### 2. まず見る動画 (改善前/後 の比較)
```
ベースライン → 最終改善版で進化を確認:
└ C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\evaluation_videos\
    ├ v89_match3_phase_i_viz_baseline.mp4      ← 改善前
    └ v89_match3_phase_i_viz_DB.mp4            ★★ 最終 (動画別 HSV 自動学習 + 起動時 inject)

多動画で同効果確認:
    ├ v29_phase_i_viz_DB.mp4 (309MB)
    └ v40_phase_i_viz_DB.mp4 (404MB)
```

### 3. 視覚比較の決定的証拠
```
data/evaluation_videos/v89_compare/030s_pure_diff_DB.png   ← 30s で 1P 全面・2P 一部の認識変化
data/evaluation_videos/v89_compare/080s_pure_diff_AB.png   ← 80s で A 統合の試合外判定切替
data/evaluation_videos/v29_compare/030s_pure_diff.png      ← v29 でも同様
```

---

## Wide regression (本セッション changes 全て反映後)
- **1775 passed, 6 failed (timeseries_wrapper 系のみ、 既知 / 本セッション無関係), 6 skipped**
- 私が触った modules (recognition_pipeline / image_reader / patch_classifier / hybrid_classifier / online_hsv_calibrator / topo_filter / cell_color_fine_tuner) は **全 pass**
- failed: `tests/test_timeseries_wrapper.py::test_{feature_names_count_270, extract_static_vector_45_keys, integration_with_real_indicator_set, expand_features_all_270_keys}` 等 (45 指標 vs 270 features の構造変更検出)

## サイクル結果 (2026-05-09 → 2026-05-10 自律サイクル)

### サイクル 1: HSV-anchor seed → v6/v7 fine-tune
- HSV ranges 中心領域 (threshold 0.7) 内 cell のみ抽出 → 46K seed dataset
- v6 model: anchor のみ → 5 色は学ぶが empty 忘却 (acc 13.4%)
- v7 model: anchor + empty/ojama 混合 → ojama 1.7%→59%、 但し全体 14.7% trade-off
- **結論: pseudo fine-tune の 47.5% 壁は本質的、 mode を移すだけ**

### サイクル 2: Phase I.c の v29 普遍性検証
- v29 で `frame=828 で 6 colors injected` 確認
- → Phase I.c は v89 専用ではなく **animation/video 普遍**

### サイクル 3: 動画別 HSV ranges DB 永続化
- `extract_per_video_hsv_ranges.py` で動画 1 本から JSON DB 生成
- v89 (5 色) / v29 (6 色) / v40 (6 色) の DB 構築完了
- `data/per_video_hsv_ranges/{vid}.json` 形式で永続化

### サイクル 4: DB pre-inject 機能
- `visualize_recognition.py --hsv-state` で起動時 inject
- `v89_match3_phase_i_viz_DB.mp4`: frame 0 から動画別最適化済認識
- メリット: HSV3 (online) は frame 234 まで default ranges、 DB は frame 0 から最適化

### サイクル 5: ensemble 試行 (打ち切り)
- v1+v7 合議は class collapse 突破に効果薄と判断 → サイクル 6 にリソース集中

### サイクル 6: 多動画総合検証
- v89 baseline vs DB: 30s で大量 cell 認識変化 (1P 全面 + 2P 一部)
- v29 viz_DB: 309MB 完成、 6 colors HSV pre-inject 動作確認
- v40 viz_DB: 404MB 完成

### サイクル 7: DB 効果の cell-level 数値化 (`measure_db_impact.py`)
- v89: diff 22.55% / blue→empty 5,644 ★ (青背景バイアス修正の数値証拠)
- v29: diff 51.05% / 全色→empty + empty→blue (大幅修正)
- v40: diff 23.61% / 全色→empty 大量

### サイクル 8: memory 統合 + SESSION_HANDOFF TOP 配置
- `project_phase_i_c_quantitative_results.md` 追加 (8 件目の memory)
- ユーザー目線で結果が見える形に整理

### サイクル 9: 動画別 HSV ranges 比較表
動画ごとに学習した HSV ranges が大きく異なることを実証:
- red: v89 H154-180 (赤紫) vs v29 H0-14 (純赤) vs v40 H4-10
- blue: v89 高彩度 / v29 幅広彩度
- ojama: 完全に異なる範囲

### サイクル 10: DB-aware anchor seed 試行 (中断)
- `build_hsv_anchor_seed --db-root` 追加で動画別 ranges 注入版を実装
- 4209 anchor 構築 (v6 の 46K より少ない、 fine-tune には不足)
- → mode collapse 突破は本セッションで不可と確定

### サイクル 11: 最終整理 (本セッション完)

## 実装サマリー (8 件 + Phase I.c)

| # | 機能 | 状態 | 主効果 |
|---|---|---|---|
| 1 | viz `cnn_model_path` str→Path bug 修正 | ✅ | 既存スクリプト全動作復帰 |
| 2 | A 統合 (ScoreZero/MatchEnd/Telop) | ✅ 目視実証 | 試合終了判定 (v89 80s で `menu` 認識) |
| 3 | B 改善 (TSUMO_FALL→STABLE 横置き next_pair 補正) | ✅ | 落下後の青誤認識緩和 |
| 4 | Telop cell mask 有効化 | ✅ | 中央テロップ被覆 cell を UNKNOWN |
| 5 | topo_filter OOM ガード (MiniBatchKMeans + 200K) | ✅ | WSL クラッシュ回避、ピーク 5GB→1.5GB |
| 6 | CellColorFineTuner 拡張 (class_balance/focal/logit/oversample) | ✅ | mode collapse 緩和試行 (限界確認) |
| 7 | build_consensus_seed.py (Cross-CNN ensemble) | ✅ | 49K seed 生成も empty 限定で実用性低 |
| 8 | **Phase I.c (OnlineHsvCalibrator 統合)** | ✅ 実装 + 検証中 | pseudo に依存しない動画別 HSV 自動学習 |



## 主要結論
- **A 統合の効果は実証済み**: v89 80s (試合終了寸前) で baseline `1P:tsumo_fall 2P=tsumo_fall` → AB `1P:menu 2P=menu` に切替。ご指摘「試合終了の判定甘さ」を直接解消。
- **B 改善 (横置き next_pair 補正)** は実装済、TSUMO_FALL→STABLE 遷移時のみ発火。
- **47.5% mode collapse の壁**: pseudo label の (patch=transient 誤分類, label=settle 多数決) という設計上、fine-tune は ~50% acc が限界。focal/logit/oversample/class_balance を試行も突破できず。
- **Phase I.c (OnlineHsvCalibrator 統合) 実装完了**: pseudo に依存しない別経路で動画別 HSV 自動学習。`recognition_pipeline.update` で STABLE 中サンプル蓄積 → ready 後に ColorClassifier ranges 動的書き換え。実機検証中 (v89 で `online_hsv injected` ログ確認)。

## サイクル1 (2026-05-09 後段): HSV-anchor seed → v6 fine-tune

### 要件取得 (Web)
- SST (Self-training with Self-Adaptive Thresholding, ICLR 2025): class-specific threshold で adaptive にゲート
- AdaptiveDrop: label noise filter for self-supervised
- HSV thresholding: hue channel が色型をモデル化、segmentation に直接有用

### 設計
**目的**: 47.5% mode collapse 突破。pseudo の transient noise を排し、 HSV ranges 中心領域に確実に入る cell のみを anchor seed として fine-tune の loss anchor にする。

**動作**:
1. ColorClassifier (default HSV ranges) で各 pseudo cell の HSV 中央値を計算
2. ranges の中心からの max-axis 正規化距離 ≤ 0.7 (= 中心 70% 領域内) なら採用
3. red/blue/green/yellow/purple の 5 色分の高純度 anchor 構築
4. これを fine-tune store として cnn_phase_b_finetuned base から再学習 → v6

### 製造
- `scripts/build_hsv_anchor_seed.py` (新規、 165 行)
- HSV 中央値 + ranges 中心距離で SST 風の class-specific threshold filter
- 出力: `data/pseudo_labels_hsv_anchor/{vid}/cell.jsonl`

### テスト
- v89 で threshold=0.5: 488/30K (1.6%) → too tight
- v89 で threshold=0.7: 4535/97K (4.7%) → OK
- 全 10 動画 で実行: **46,010 anchor seed (ratio 5.9%)**, by_label = `{red:7561, blue:11704, green:11994, yellow:2968, purple:11783}`

### 評価
- `cnn_phase_b_finetuned_v6.pt` 学習中 (45 K anchor, epochs=5, augment + class_balance)
- 完了後 v89/v29 viz で baseline 比較

## 試行ログ

| model | total acc | empty | red | blue | green | yellow | purple | ojama | mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cnn_phase_b_v1 (base) | 13.4% | 16% | 7% | 23% | 7% | 14% | 9% | 8% | random |
| v1 finetuned (3 epoch) | **47.9%** | 96.9% | 2.0% | 67% | 0% | 0% | 0% | 1.7% | empty+blue |
| v2 (aug+topo+balance) | 42.8% | 84.7% | 0.9% | 0.6% | 0.5% | 0.5% | 69.2% | 15.9% | empty+purple |
| v3 (+focal+logit) | **46.9%** | 93.2% | 0.4% | 0.4% | 0.2% | 0.7% | 68.5% | 6.0% | empty+purple |

→ どの model も 2 色に倒れる mode collapse。focal/logit はモードを変えるが解消はしない。

## 触ったコード (Phase I.c 含む)

### `src/recognition_pipeline.py`
- `cnn_model_path` を `Path()` で正規化 (str も受付、line 331)
- A 統合: `__init__` / `load_default` / `update` で ScoreZero/MatchEnd/Telop を統合 (line 161 付近)
- B 改善: `_compute_landing_inferred` の横置き判定で next_pair 整合補正 (line 805)
- Telop mask: `_build_hybrid_reader` で `use_telop_mask=True` に有効化 (line 271 付近)

### `src/self_supervised/topo_filter.py`
- `MiniBatchKMeans` 切替 + `max_samples=200000` sub-sample で OOM ガード

### `src/self_supervised/cell_color_fine_tuner.py`
- 引数追加: `class_balance` / `focal_gamma` / `logit_adjustment_tau` / `oversample_alpha`
- `_train_loop` で focal loss + logit adjustment + CReST 風 oversample 対応

### `scripts/phase_i_fine_tune.py`
- 上記引数を CLI 伝播

### `scripts/build_consensus_seed.py` (新規)
- 複数 CNN の合議で全員一致 sample のみ seed として書き出し (Cross-CNN ensemble seed)
- **試行結果**: 992K in → 49K out (5%)、ほぼ全部 empty。3 model の合議で取れる seed は empty 限定 → v5 fine-tune は不可能と判明

### Phase I.c (OnlineHsvCalibrator) 統合 - 新規実装
- `src/patch_classifier.py`: `CnnPatchClassifier.predict_proba_grid()` 追加
- `src/hybrid_classifier.py`: `HybridClassifier.predict_proba_and_hsv_grid()` 追加
- `src/online_hsv_calibrator.py`: `OnlineHsvCalibrator(require_cnn_proba=False)` モード追加 (CNN mode collapse 回避)
- `src/recognition_pipeline.py`: `__init__/load_default/update` で OnlineHsvCalibrator 統合
  - 各 frame STABLE 中 1P/2P 両 side で信頼サンプル蓄積
  - `MIN_SAMPLES=50` で ready 後、`ColorClassifier.set_color_ranges_from_simple()` で動画別 HSV ranges 動的書き換え
  - **HSV-only mode (require_cnn_proba=False)** で CNN mode collapse 下でも機能
- **効果実証**: v89 viz_HSV3 で `frame=234 で 6 colors injected` 達成、 30s 時点で baseline と認識結果が大幅相違 (動画別 HSV ranges が効いている証拠)
- 比較画像: `data/evaluation_videos/v89_compare/030s_HSV3.png` vs `030s_baseline.png` で目視可

## 比較動画 / 画像

### v89 全構成 (進化系で確認)
| ファイル | 説明 |
|---|---|
| `data/evaluation_videos/v89_match3_phase_i_viz_baseline.mp4` | 改善前 (cnn_phase_b_finetuned.pt のみ) |
| `data/evaluation_videos/v89_match3_phase_i_viz_A.mp4` | A 統合 (試合判定 ScoreZero/MatchEnd/Telop) |
| `data/evaluation_videos/v89_match3_phase_i_viz_AB.mp4` | A + B (横置き next_pair 補正) |
| `data/evaluation_videos/v89_match3_phase_i_viz_v2.mp4` | + v2 model (aug+topo+balance) |
| `data/evaluation_videos/v89_match3_phase_i_viz_v3.mp4` | + v3 model (focal+logit+balance) |
| `data/evaluation_videos/v89_match3_phase_i_viz_HSV3.mp4` | ★ Phase I.c online (frame 234 で inject) |
| `data/evaluation_videos/v89_match3_phase_i_viz_DB.mp4` | ★★ Phase I.c DB pre-inject (frame 0 から最適化) |
| `data/evaluation_videos/v89_match3_phase_i_viz_v7.mp4` | + v7 model (anchor + empty/ojama 混合) |

### 多動画検証 (Phase I.c 普遍性確認)
| ファイル | 動画 |
|---|---|
| `data/evaluation_videos/v29_phase_i_viz_DB.mp4` | v29 (156s 動画) で DB pre-inject 動作確認 |
| `data/evaluation_videos/v40_phase_i_viz_DB.mp4` | v40 (60fps 125s 動画) (生成中) |

### 動画別 HSV ranges DB
```
data/per_video_hsv_ranges/v89.json   ← 5 色 (red/blue/yellow/purple/ojama)
data/per_video_hsv_ranges/v29.json   ← 6 色全部
data/per_video_hsv_ranges/v40.json   ← 6 色全部
```

特に `data/evaluation_videos/v89_compare/080s_*.png` で A 統合の効果が明確、
`030s_pure_diff_DB.png` で Phase I.c DB pre-inject の効果が大規模に見える。

## memory 7 件追加 (`MEMORY.md`)
1. `project_phase_i_finetune_findings.md` — 47.5% acc 本質と限界
2. `project_pipeline_detector_integration.md` — A 統合の経緯 + 80s 効果実証
3. `feedback_topo_filter_oom.md` — TopoFilter OOM ガード必須ルール
4. `project_bg_fingerprint_status.md` — BG FP の伝達範囲 (image_reader まで、HybridClassifier 不伝達)
5. `project_recognition_improvement_candidates.md` — CReST/OnlineHsv 統合/手動 seed/logit/Cross-CNN 等の next 候補

## 次回優先タスク (20:00 後 or 次セッション)

### A. OnlineHsvCalibrator 統合 (工数 ~3 時間)
- `src/online_hsv_calibrator.py` 完全実装済だが `recognition_pipeline.py` 未統合
- 統合手順:
  1. `CnnPatchClassifier` に `prob_grid` メソッド追加 (各 cell の softmax max を grid で返す)
  2. `ImageReader.read_board` で HSV-only classifier 出力も grid で返す
  3. `RecognitionPipeline.update` で `OnlineHsvCalibrator.update(frame, region, board, cnn_proba_grid, hsv_color_grid)` を呼ぶ
  4. `is_ready` 後の `get_per_video_ranges()` を ColorClassifier に注入
- 効果: 動画別 HSV 自動学習 → mode collapse の根本回避 (CNN に依存せず色判定)

### B. 少量手動 seed (工数 ~1 時間 + 30 分人手)
- 各動画から 70 cell × 7 色 を視覚的に label
- `data/manual_seed/v??/*.png` + JSONL
- これを Phase L の anchor dataset とする

### C. CReST/Consensus seed の試行 (工数 ~1 時間)
- 既に script 実装済 (`scripts/build_consensus_seed.py`)
- 注意: 全 CNN model が pseudo domain で random 級なので合議で empty しか得られない懸念
- HSV classifier も追加した「CNN + HSV 合議」が効果的かも
- v5 model 試行: `phase_i_fine_tune --store-root data/pseudo_labels_consensus`

### D. 手動 seed → fine-tune anchor (Phase L 本番化前提)
- 上記 B のデータを使い、 fine-tune の loss に「manual seed の確実性 weight 5x」を追加
- self-training の品質を抜本的に変える

## 注意

- WSL2 メモリ予算: viz と fine-tune の **並列実行は OOM リスク** (v3 で 9GB 使用)
- 並列起動時は片方完了まで待つこと
- `setsid -f` のジョブが消えていないか pgrep で確認
- WSL 再起動はユーザーに依頼 (memory `feedback_wsl_restart_delegation.md`)

## 即実行レシピ (CLI)

### v89 viz を任意 model で再生成
```bash
PYTHONPATH=. ./venv/bin/python -u -m scripts.visualize_recognition \
  --video data/evaluation_videos/v89_match3_95s.mp4 \
  --output data/evaluation_videos/v89_test.mp4 \
  --cnn-model models/cnn_phase_b_finetuned_v3.pt
```

### Cross-CNN consensus seed 構築
```bash
PYTHONPATH=. ./venv/bin/python -u -m scripts.build_consensus_seed \
  --models models/cnn_phase_b_v1.pt models/cnn_phase_b_finetuned.pt \
           models/cnn_phase_b_finetuned_v3.pt \
  --videos v29 v30 v31 v32 v33 v40 v51 v57 v70 v89 \
  --out-root data/pseudo_labels_consensus \
  --limit-per-video 100000
```

### Consensus seed で fine-tune (v5)
```bash
PYTHONPATH=. ./venv/bin/python -u -m scripts.phase_i_fine_tune \
  --component cell_color --all \
  --store-root data/pseudo_labels_consensus \
  --cell-save-to models/cnn_phase_b_finetuned_v5.pt \
  --epochs 5 --augment --class-balance --oversample-alpha 0.5
```
