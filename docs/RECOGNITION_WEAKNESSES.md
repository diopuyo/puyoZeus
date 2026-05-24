# 認識スタック弱点リスト

**更新**: 2026-05-09 (PM、先行研究 web 調査追記)
**目的**: Phase I 実装漏れ (cell 色 CNN) を踏まえて、認識スタック全体の弱点を網羅的に洗い出し、各項目を独立した task として個別判断可能にする。

## 既知 / 着手済

| 弱点 | 状態 | Task |
|---|---|---|
| Cell 色 CNN 未 fine-tune | Phase I.b 実装中 | #28 |
| EffectPhaseDetector skeleton (常に None) | 未着手 | #29 |
| OnlineHsvCalibrator 未統合 | memory 既知、未 task 化 | - |
| Hidden row Platt scaling のみ | Phase I 部分対応 | (済) |

## 新規発見 / 軽視されていた弱点

### 認識層 (Recognition)

#### R-1: TSUMO_FALL 検出の緩さ
**現象**: ぷよ落下中の puyo を捕捉できず STABLE 確定が遅れる/抜ける
**影響**: 9 秒の 2 puyo 未認識の根本原因の一部
**対策**:
- TsumoFallDetector の精度向上
- 自己整合性 (next 落下後の盤面色 count delta vs next_pair 期待) で validate
**工数**: 2-3 日

#### R-2: Class imbalance (EMPTY が cell の 80%+)
**現象**: cell CNN が EMPTY ばかり学習、minor class (紫等) under-represented
**影響**: 近似色 (紫/青、黄/緑) の誤認固定化
**対策**:
- weighted loss / focal loss
- over-sampling minor class
- balanced batch sampler
**工数**: 1-2 日 (CNN 学習 pipeline 変更)

#### R-3: 物理整合性チェックの不在
**現象**: ぷよぷよルール (4+連結消去、重力、色種数 ≤5) を pseudo-label に適用しない
**影響**: bootstrap で誤認永続化リスク (例: 紫を青と settle → 青で学習 → 永続)
**対策**:
- PhysicalConsistencyValidator: 各擬似ラベルがぷよぷよルールに整合しているかチェック
- 不整合 sample は学習データから除外 or low confidence
- 全 validator (score / next / chain / cell_color / hidden_row) に共通 hook
**工数**: 3-5 日 (cross-cutting concern)

#### R-4: Triple next (3 手目) 未対応
**現象**: 上級者は 3 手先を読む、視認可能な情報 (3 つ目の next ROI) を pipeline で読まない
**影響**: 戦況評価で限定的な可視情報
**対策**:
- TripleNextDetector 新規 (next_detector を拡張)
- 自己整合性 (1 frame 後に dnext → next スライド時、triple_next → dnext へ)
**工数**: 2-3 日

#### R-5: 解像度 edge case
**現象**: 720p で ROI 全早期 return の経験 (Phase I.b 直前で発覚)。480p、4K、非標準 aspect 未検証
**影響**: 別解像度動画で全 frame 認識失敗
**対策**:
- 動画読込時に必ず 1920×1080 に正規化 (一部スクリプトで実装済)
- 全 ROI 系 module で `_ensure_1080p` を統一
- 解像度別自動 ROI 再キャリブ (将来)
**工数**: 1 日 (統一)、5-7 日 (本格自動キャリブ)

#### R-6: Player overlay / 解説テロップ干渉
**現象**: 試合中の overlay (プレイヤー名、レート、解説テロップ) が盤面に被ると誤認
**影響**: 大会動画でテロップ重なる時間帯で認識破綻
**対策**:
- UI mask の動的拡張 (template matching で overlay 領域検出)
- 解説者名動画で Robustness 検証
**工数**: 3-5 日

### Self-supervised 層 (Phase I 実装品質)

#### S-1: score_validator の桁分解問題
**現象**: score 値を 8 桁ゼロ埋めで分解、各桁の picture は位置依存だがラベルは値依存
**影響**: 各桁の見た目 (フォント・大きさ) が異なるので学習混乱
**対策**:
- 各桁 patch を独立 sample として保存 (位置情報も保持)
- 桁別の digit テンプレ更新
**工数**: 1-2 日

#### S-2: next_validator lenient confidence 0.85
**現象**: 「3 frame 連続観測」で MEDIUM 0.85 emit、信頼性低いデータも採用
**影響**: fine-tune 汚染で精度劣化リスク
**対策**:
- confidence 0.95 のみで fine-tune ablation 検証
- lenient mode は別 mode で隔離
**工数**: 0.5-1 日

#### S-3: chain_validator が deterministic で no-op
**現象**: ChainSimulator が deterministic なので chain detection 既に 100% 一致 = 改善余地なし
**影響**: 11K samples が宝の持ち腐れ、本当に 100% 正確かも疑問
**対策**:
- 別 ground truth source (動画 manual annotation 数 sample) で検証
- もし誤りある = chain detect 改善余地あり
**工数**: 1 日 (manual annotation)、+ 改善次第

#### S-4: bootstrap 自己強化問題 (R-3 と関連)
**現象**: 誤認が settle して訓練データに混入 → 誤認固定化
**影響**: cell color 学習で「紫を青と覚える」モード固定化リスク
**対策**:
- 物理整合性 (R-3) との交差検証
- 多視点クロスチェック (HSV 単独判定 + CNN 判定の 2-of-3 majority)
- 学習前 / 学習後で bootstrap drift 監視
**工数**: 3-5 日

#### S-5: Cell 色 settle pattern の落とし穴
**現象**: 5 frame 連続同色 = settle と判定、しかし「5 frame 連続同色誤認」も実在 (例: 13 秒の 2 秒紫→青誤認)
**影響**: 同上 (S-4)、bootstrap 永続化
**対策**:
- 窓を長く (例: 10 frame 連続)
- 物理整合性チェック
- HSV 単独判定との 2-of-3 majority
**工数**: Phase I.b 内で対処推奨 (1-2 日)

### Pipeline / Infrastructure

#### P-1: 依存関係連鎖 (cascade failure)
**現象**: score OCR 失敗 → chain detect 失敗 → 評価データ失敗
**影響**: 1 component の破綻が pipeline 全体に伝搬
**対策**:
- 各 component に独立 validator + 縮退モード (degraded mode、警告のみで処理継続)
- DAG 化された依存関係 documentation
**工数**: 5-7 日

#### P-2: 試合境界検出の他動画未検証
**現象**: count_match_v4/v5 は video_02 のみ完全一致確認
**影響**: 他動画での境界誤検出が学習データ全体を汚染
**対策**:
- 各 Top-tier 動画で境界精度自己検証 (count_match 結果を視覚化)
- 改善版 count_match_v6 検討
**工数**: 2-3 日 (検証)、+ 改善次第

#### P-3: stream_overlay (Phase J) 未検証
**現象**: OBS 統合動作未確認、最終 deliverable に影響
**影響**: ローンチ直前で発覚すると致命的
**対策**:
- 早期試験運用 (Phase L 並行で OBS 環境準備)
**工数**: 2-3 日 (Phase J で実施)

#### P-4: per_video_model_selector ハードコード
**現象**: v17b/v16 の動画別 mapping は固定 (memory `cnn_phase_b_v1` 等)、新動画で自動決定無し
**影響**: 動画追加時に毎回手動チューニング
**対策**:
- 自動 selector (新動画で各 model holdout 評価 → 最良採用)
- per_video_hsv_stats.json を活用 (memory `realtime_hsv` 段階 3)
**工数**: 3-5 日

### 時系列 / モデル層

#### M-1: cnn_phase_b_v1 表現力不足
**現象**: 25KB と極小、紫/青 や 黄/緑 のような近似色を区別しきれない可能性
**影響**: cell 色誤認の根本原因の一部
**対策**:
- より大きな CNN (100K-1M params) で fine-tune
- Phase L 用 `board_cnn_pretrained.pt` 起点で transfer learning
**工数**: 1 週間 (CNN 設計 + データ + 学習 + 検証)

#### M-2: augmentation の限定性
**現象**: 既存学習データは特定動画群から、新 UI で破綻
**影響**: 大会別 / プレイヤー別の頑健性低下
**対策**:
- color permutation, JPEG noise, brightness/contrast jitter, rotation, crop
- 学習 pipeline に統合 (現状ほぼ素のまま)
**工数**: 2-3 日

#### M-3: temporal_smoother (旧) 未使用
**現象**: 新 pipeline は smoothing=1 = 平滑化なし、1-frame ノイズ脆弱
**影響**: 一瞬誤認の Phase I 改良が temporal 層では補強されてない
**対策**:
- smoothing window 拡張 (例: 3-5 frame)
- または stable_frame_count 増 (現 6 → 8 等)
**工数**: 1 日 + 影響評価

## 先行研究由来の追加項目 (2026-05-09)

### R-7: NEXT window slide motion 検出 (TsumoFallDetector 補強)
**現象**: next 1 → next 0 へのスライドモーション = 確実に「手が打たれた」signal だが未利用
**影響**: R-1 (TsumoFallDetector) と組み合わせで 9 秒問題 (2 puyo 未認識) に効く
**出典**: PuyoSpectatorAssist の改善計画
**対策**:
- next ROI の差分検出 (previous frame との pixel-wise diff)
- diff > threshold = slide motion = 手打ち確定 → state 遷移補強
**工数**: 1-2 日 (R-1 と統合実装)

### B-1.b: Sullen GTR + Fron 形テンプレ追加
**現象**: 上級者の追加形パターン (Sullen GTR、Fron) が未実装、form 指標 4 個のみ
**影響**: 戦術評価で特定形の使用率が捕えられない
**出典**: citrus610/ama (C++ AI、3 形パターン: GTR、Sullen GTR、Fron)
**対策**:
- `src/form_templates.py` に SullenGtrTemplate、FronTemplate 追加
- 既存 GTR / LLR / 階段 / 座布団 と同パターン (等価クラス、1P/2P mirror)
- `EXTRA_INDICATOR_NAMES` に form_sullen_gtr、form_fron 追加 (45 → 47)
**工数**: 2-3 日

### M-4: ONNX export + OpenCV DNN inference (CNN 大型化後の高速化)
**現象**: M-1 で CNN 大型化したらリアルタイム推論で FPS penalty
**影響**: Phase J オーバーレイの latency 200ms 目標達成困難
**出典**: puyogg/puyo-classification (PyTorch → ONNX で MLP は CNN より速い前提)
**対策**:
- M-1 完了後、PyTorch → ONNX export
- OpenCV DNN または onnxruntime で推論
- batch 効率や量子化検討
**工数**: 1-2 日 (M-1 完了後)

## 先行研究 web 調査由来の追加項目 (2026-05-09 PM)

調査範囲: GitHub (ぷよぷよ専門 / Tetris CV / Match-3)、Qiita / はてな / J-STAGE 論文、arxiv (TTA・self-supervised)。
詳細は memory `reference_puyo_ai_recognition.md` にも追記済。

### R-8: 形状ベースのテンプレートマッチング (色不変認識)
**現象**: HSV+CNN は色シフトに弱い (動画別キャリブが必要)。色変動の大きい大会動画
(派手なエフェクト・ライティング変更・配信フィルタ) で破綻リスク。
**影響**: M-2 (augmentation) で大半カバー想定だが、未知大会動画で R-2 / S-4
クラスのバイアスが残る。
**出典**: bpinzone/TetrisAI (Tetris 99 で世界 1 位達成、HDMI 取得)。
- 色は環境依存で失敗 → **形状を白黒マスク化してテンプレートマッチング**
- 「95% 以上一致」を要求して連鎖エフェクト (sparkle) を排除
- 浮遊ブロックを **BFS で底面非接続を検出して除外** (連鎖中の落下中ぷよ ≠ confirmed)
**対策**:
- ぷよ 1 個の "shape mask" (円形 + 目玉) を雛形化、グレースケール二値化との
  IoU を補助スコアとして HybridClassifier に追加
- 「色が判定できなくても形は puyo らしい」セルを `COLOR_UNKNOWN` で安全に保持
- BFS 浮遊検出は ProbabilisticBoard と組み合わせて settle 候補を厳格化
**工数**: 3-5 日 (shape mask 生成 + IoU スコア統合)
**ライセンス**: bpinzone/TetrisAI 個人ブログ、明示 license なし → **手法のみ参照** (実装は再構築)

### R-9: 「次フレーム予測へフォールバック」戦略 (連鎖エフェクト時の安全網)
**現象**: 全消し / 連鎖中の派手エフェクトで cell 認識が大量に汚染される。
現状は EffectPhaseDetector が常に None で、当該 frame は最後 STABLE で凍結する
ものの「次に STABLE になる時の盤面」予測は無し。
**影響**: 36 秒問題 (memory `feedback_chain_phase_physics_only.md`) が物理推論で
部分的に解決済だが、エフェクト後の SETTLE 検出遅延は残存。
**出典**: bpinzone/TetrisAI 「目を閉じる戦略 (close eyes)」=
*「次フレーム予測の方が sparkle でぐちゃぐちゃの観測より信頼できる」*
**対策**:
- ChainSimulator を frame レベルで予測 (STABLE → action → 連鎖 → next STABLE)、
  連鎖中は予測盤面を表示し、SETTLE 検出 frame で観測と突合
- 観測 vs 予測の差分 > threshold → 警告出力 (cascade failure 検出)、ただし
  cumulative bias 監視に活用
- EffectPhaseDetector skeleton (#29) の本格実装と組合せ
**工数**: 5-7 日 (ChainSimulator の frame 単位化 + 突合ロジック)
**ライセンス**: 同上 (手法のみ)

### R-10: Match-3 系で実証済「グリッド検出 → セル分割」のロバスト化
**現象**: 現状 ROI は固定座標、解像度調整は 1920×1080 想定。R-5 で挙げた
edge case 対応が薄い。
**影響**: 解像度 / aspect 比異なる動画 (海外配信、smartphone capture 等) で
全 frame 認識失敗の経験あり。
**出典**:
- daniel-bandstra/watchGo (Go board OpenCV、Canny + Hough Line で grid 検出)
- match-3 系 OpenCV Q&A (Adaptive thresholding でグリッド線抽出)
- chess board recognition 系 repo (corner detection + perspective transform)
**対策**:
- `BoardGridDetector` 新規: Canny + Hough Line + 交点クラスタリングで
  6×13 グリッド四隅を自動推定
- 動画読込時に 1 回キャリブして以降キャッシュ (per-video)
- 失敗時は固定 ROI fallback
**工数**: 4-6 日 (検出 + 評価 + fallback 統合)
**ライセンス**: 各 repo MIT 系 (watchGo は明示なし、手法のみ参照)

### S-6: テンプレートマッチング score OCR (Tesseract 不要)
**現象**: S-1 で score 桁分解の問題を挙げた。Tesseract は静的フォントで
**「精度フラストレーションが大きい」** という共通報告 (NES OCR 議論、
Video-Game-OCR repo issue 等)。
**影響**: score validator の confidence が見かけより低く、Phase I の
self-validation 全体に波及。
**出典**:
- leshokunin/Video-Game-OCR (Tesseract で confidence 設定可能だが game UI で limit)
- NES OCR 議論 → **「digit テンプレート pixel-level 比較が安定」**が結論
- BitterOcean/Digit-Recognition-Using-Tesseract (Tesseract は学習で改善するが
  custom font は専用 .traineddata 必須)
**対策**:
- ぷよぷよ score 領域から数字 0-9 の binary template を 1 回 capture (動画別)
- 各桁 patch を template との pixel diff で分類、Tesseract は fallback のみ
- 既に持つ digit テンプレと統合 (S-1 と同時実装)
**工数**: 1-2 日 (S-1 改善と同時)
**ライセンス**: leshokunin/Video-Game-OCR は明示なし、Tesseract 4 BSD-like

### M-5: Test-Time Adaptation (TTA) — 未知動画への自動適応
**現象**: OnlineHsvCalibrator は HSV パラメータのみ動画別調整。CNN 重みは固定で
未知動画への adaptation なし。
**影響**: Phase L で動画 100-150 本に拡大すると、各動画の skin/light 差が
HSV だけでは吸収しきれない可能性。
**出典**:
- Sun et al. *"Test-Time Training with Self-Supervision for Generalization
  under Distribution Shifts"* (2019)
- TTT++ (NeurIPS 2021) — auxiliary task の loss を test 時にも minimize
- *Video Unsupervised Domain Adaptation* survey (ACM Computing Surveys 2024)
- awesome-test-time-adaptation (tim-learn/) — 手法カタログ
**対策**:
- Phase I の self-validation を **TTA の auxiliary supervision** として再利用
  (chain consistency / score consistency / color count delta が test loss)
- 動画開始 30 秒で BatchNorm statistics を更新 (TENT 方式、1 epoch fine-tune)
- adaptation 前後で ablation 検証
**工数**: 1-2 週間 (M-1 大型 CNN 完了後、TTA layer の追加 + 検証)
**ライセンス**: 手法は学術 papers (license 制約なし)、実装は再構築

### M-6: WB color augmenter (動画別 white balance ドリフト対応)
**現象**: M-2 で augmentation 工数化済だが、white balance shift augment は
未明示。配信プラットフォーム (Twitch / YouTube) の auto WB が動画別に異なる。
**影響**: 動画別の彩度・色温度差で誤認固定化リスク (S-4 と関連)。
**出典**: Afifi et al. *"WB color augmenter"* (ICCV 2019、MIT license)
- 大会動画でも有効 (illumination class shift augmentation で +2-5pt 報告事例)
- mahmoudnafifi/WB_color_augmenter (Python + Matlab 実装)
**対策**:
- 既存 augmentation pipeline に WB shift を追加 (色温度 ±2000K、tint ±)
- M-2 と同時実装で工数追加なし
**工数**: 0.5-1 日 (M-2 の追加 augment として)
**ライセンス**: MIT (採用 OK)

### S-7: 物理整合性ベースの噪声フィルタ (Match-3 系の経験則)
**現象**: R-3 で物理整合性チェックを挙げたが、具体的にどう pseudo-label に
適用するかは未設計。
**影響**: bootstrap 自己強化の根本対策。
**出典**:
- TopoFilter (NeurIPS 2020、`pxiangwu/TopoFilter`)
  *"A Topological Filter for Learning with Label Noise"*
  → noise level >= 0.8 でも k-cluster + majority voting で外れ値除去
- match-3 系 OpenCV 議論で「色 ≠ 形」分離の重要性
**対策**:
- pseudo-label を connected component で grouping (色不問)
- 同 component 内で **「色順序対称 4!=24 で集約」** (添島・山口 2019 の知見)
- 多数決で代表色を確定、少数派 cell は除外 or low confidence
**工数**: 3-4 日 (pseudo-label pipeline に統合)
**ライセンス**: TopoFilter は学術 (実装は public)、手法のみ参照

### B-2: 色順序対称性によるデータ augmentation (添島・山口 2019)
**現象**: ぷよぷよでは「赤+青の連鎖」と「黄+紫の連鎖」は **意味的に同一**。
現状 CNN/MLP は各色を独立学習、データ効率悪い。
**影響**: 学習データを 24 倍に水増し可能 = Phase L 時の必要動画数削減。
**出典**: 添島・山口 「深層学習を用いたぷよぷよ AI の開発」 (FSS35、2019、佐賀大)
- 4 色の permutation で 4! = 24 通りの「等価盤面」生成
- 1 つの盤面データで 24 状態を表現、学習効率 ×24
**対策**:
- `src/color_permutation_augment.py` 新規: cell label を 24 通り permute
- cell color CNN 訓練時に runtime で random permute (memory 節約)
- M-2 augmentation pipeline に統合
**工数**: 1-2 日 (cell color CNN 訓練 loop に runtime augment 追加)
**ライセンス**: 学術 paper (J-STAGE 公開、手法のみ参照)

### M-7: UPI protocol への対応 (CV 出力 → AI engine I/F 標準化)
**現象**: 現状 CV 出力は内部 dict / numpy のみ。citrus610/ama や
TukamotoRyuzo/upi-protocol が標準化を進めている。
**影響**: Phase J オーバーレイ後、AI engine と組合せて「分析ツール」化したい時に
独自 protocol だと採用されない。
**出典**:
- TukamotoRyuzo/upi-protocol (UCI 模倣、`pfen` 文字列で盤面送信)
- citrus610/ama, frostburn/puyobot が UPI 対応
**対策**:
- `src/upi_emitter.py` 新規 (Phase L 後): confirmed_board → pfen 変換 +
  next pair (tumo) emission
- 既存 ChainSimulator の出力は ama / puyobot で再利用検証
**工数**: 2-3 日 (Phase L 後の standalone task)
**ライセンス**: upi-protocol 明示なし、puyobot は MIT (連鎖シミュ参考可能)

## 採用優先度サマリ (本セッション追加分)

**Top 3 (即効性 + 実装軽量):**
1. **B-2 色順序対称 augment** (1-2 日、Phase I.b の cell CNN 訓練に直接効く)
2. **M-6 WB color augmenter** (0.5-1 日、MIT 採用可、Phase L 一般化に直結)
3. **S-6 template matching score OCR** (1-2 日、S-1 と同時、Tesseract 信頼問題回避)

**中優先 (Phase L 周辺で検討):**
- R-10 グリッド自動検出 (解像度 robustness、4-6 日)
- M-5 TTA (M-1 大型 CNN 後、1-2 週間)
- S-7 物理整合性 noise filter (R-3 の具体実装、3-4 日)

**低優先 (本格運用後):**
- R-8 形状テンプレート (HSV+CNN で十分なら不要、3-5 日)
- R-9 次フレーム予測フォールバック (5-7 日、EffectPhaseDetector 完成後)
- M-7 UPI protocol (Phase L 完了後、2-3 日)

## 教訓 (Phase I の反省)

**「自己学習可能なものすべて」の網羅原則**:

Phase I 当初、ユーザーが指摘した score / next / chain / hidden_row は実装したが、
**最も影響大な cell 色 CNN は抜け漏れ**。今後の原則:

1. **画像認識可能なすべての出力** に対して self-validation を 1 つは設計
2. **物理整合性** (ぷよぷよルール) で全 component を交差検証
3. **依存関係** を mapping して cascade failure 回避
4. **Phase 着手前に網羅チェック**: 「指示された項目以外で抜けているものはないか?」

## 推奨着手順 (参考)

**緊急 (Phase I.b 完了直後):**
- R-3 物理整合性チェック (bootstrap 防止、全 validator に効く)
- S-5 cell color settle pattern 強化 (Phase I.b 内で対処すべき)
- R-1 TsumoFallDetector (9 秒問題、Phase I.b で部分解消するが本格対応要)

**中期 (Phase L 前):**
- EffectPhaseDetector 本格実装 (#29、36 秒問題)
- OnlineHsvCalibrator 統合
- M-1 CNN 大型化 (board_cnn_pretrained.pt 活用)

**Phase L 並行:**
- R-4 Triple next、R-5 解像度 edge case、R-6 UI 干渉
- M-2 augmentation、M-3 temporal smoothing
- P-2 試合境界検証、P-3 stream_overlay 検証、P-4 per_video selector 自動化

## 関連参照

- `docs/IMAGE_RECOGNITION_OVERVIEW.md` — 認識スタック完全説明
- `docs/INDICATOR_ROADMAP.md` — Phase H/I/J/K/L ロードマップ
- `data/verify/phase_i_dashboard.md` — Phase I 結果
- `data/verify/learning_impact_audit.md` — 学習影響 audit
