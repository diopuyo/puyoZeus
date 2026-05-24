# 未知動画 53s 物理推論不全 レポート (2026-05-11)

ユーザー報告: `unknown_viz_FIX_E.mp4` 53s 時点で 1P puyo 7 割未認識、 連鎖後も認識復帰しない。

## TL;DR

主因 3 つ:
1. **TelopDetector の template 不在** — 動画中央の「新おいリーグ チャレンジャー決定戦 30先」テロップを検出できず、 state が STABLE のまま被覆 cell を空判定
2. **動画解像度 360p (= 既知動画の 1/4)** — 通常 720p+ で訓練、 360p だと puyo cell が ~30×30px しかなく CNN/HSV ともに精度低下
3. **HSV ranges が `_merged_default.json` (= 多動画 union)** — 動画固有適合がなく false-positive/false-negative 両方多発

## 詳細調査結果

### 検証 1: TelopDetector 動作確認 (主因)
```
t=50s telop_visible=False
t=53s telop_visible=False
t=55s telop_visible=False
t=60s telop_visible=False
```

**期待**: 「30先」バナー表示中は `effect_visible=True` → state=EFFECT → confirmed_board freeze
**実態**: バナーが盤面の row 5-8 を覆っているのに `telop_visible=False` のまま → state=STABLE → image_reader が被覆 cell を再読み込み → 空判定 → confirmed_board に永続的に空が書き込まれる

**根本原因**: `models/ui_templates/` に `telop_challenger.png` のみ。 「新おいリーグ チャレンジャー決定戦」 「30先」 等の他バナー template 未登録。

### 検証 2: 解像度ミスマッチ (副因)
| 動画 | 解像度 | fps |
|---|---|---|
| v89_match3_95s | 1280×720 | 30 |
| v40_match7_125s | 1280×720 | 60 |
| v29_match2_156s | 1280×720 | 30 |
| **unknown_match_120s** | **640×360** | 30 |

`image_reader` は 1920×1080 にリサイズ前提だが、 360p → 1080p は 3 倍 upscale で puyo の境界がぼやける。 1080p 訓練 CNN にとって patch quality が低下。

### 検証 3: 物理推論機構の現状
- 連鎖中・落下中は `BoardStateMachine` が confirmed_board を freeze (memory `chain_phase_physics_only`)
- ただし state==STABLE 中は毎 frame `image_reader` の出力で confirmed_board を更新
- → 静止 telop による被覆では state は STABLE 継続 → freeze せず empty で上書きされる

`_compute_landing_inferred` (B 改善) は 1 ツモ落下の直後に 2 cell を next_pair で補正するのみで、 既存 puyo の維持は対象外。

## 改善案 (優先順)

### A. **緊急**: TelopDetector の template 拡充 (工数: 30 分)
1. `data/test_unknown/compare/053s_RAW.png` から「新おいリーグ チャレンジャー決定戦 30先」 banner 領域を切り出し
2. `models/ui_templates/telop_challenge_league.png` 等として保存
3. 自動再ロード — TelopDetector は dir scan で全 `telop_*.png` を template として読む

### B. **本質的**: cells_covered の image_reader 統合 (工数: 1-2 時間)
TelopDetector は既に `cells_covered(region, frame_shape)` を実装済 (= memory `project_recognition_improvement_candidates` の C2 残課題)。 image_reader 1st pass に挿入して該当 cell を強制 COLOR_UNKNOWN に → ProbabilisticBoard が前回の確率分布を維持 → telop 終了後も既存 puyo を保持。

### C. **長期**: 解像度別パイプライン (工数: 大)
- 360p 入力時は CNN 信頼度を下げ、 HSV 主導 + ROI 拡張 (cell 周辺 1px 余白を取る)
- もしくは 360p 動画は学習データから除外、 720p+ のみ対象とする scope 確定
- (最終的に Phase L 本番化前にデータ filter で 720p+ 強制が現実解)

### D. **補助**: 物理推論で「消えていない puyo」 を維持 (工数: 中)
連鎖直後・telop 表示直後に、 confirmed_board を「全て新規認識結果に置換」 ではなく 「new_color が UNKNOWN なら旧色維持」 ロジックに変更。 ProbabilisticBoard でこれを実現済だが、 確定盤面層で同等の logic を入れる。

## 推奨アクション

ユーザー意向次第だが、 推奨順:
1. **A → B**: 30 分 + 1-2 時間 = 当日中に解決可能。 telop 種類が増えるたびに template 追加するのは手作業だが効果大
2. **D**: 副次的補強として、 telop 検出に依存しない頑健性を上げる
3. **C**: 解像度フィルタの方が「品質保証」 として有効、 ただし学習対象動画減少と引き換え

## 関連 memory

- `project_recognition_improvement_candidates.md` (C2: Telop cell mask 統合)
- `project_pipeline_detector_integration.md` (TelopDetector 統合状態)
- `feedback_chain_phase_physics_only.md` (STABLE 以外で CNN 信用しない)
