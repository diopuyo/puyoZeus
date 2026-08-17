# 試合境界マルチシグナル + STABLE持続確認 統合設計 (2026-08-17、アーキ)

工程上の位置づけ (user承認 2026-08-17): W25クローズ → ②判定報告 → **本設計の実装** → 148再収集再開 → Phase J。
動機となる定量事実: STABLEスナップショット汚染40% (下限)、綺麗な静止盤面だけなら99.93%
(`data/verify/board_labels_general_2026-08-17/RESULTS.md`)。

## 前提の訂正 (最重要)
本設計は「ゼロから境界マルチシグナルを作る」ものではない。既存資産:
- `src/match_end_detector.py::MatchEndDetector` — やった/ばたんきゅーテンプレNCC (`models/ui_templates/match_end_{yatta,batan}.png`)、閾値0.55、lockdown 5.0秒
- `src/match_state.py::MatchStateDetector` — 背景輝度ヒューリスティック (IN_MATCH_V_MAX=170)
- `src/match_winner.py::MatchWinnerDetector` — WIN★パネル数字ハミング距離差分で勝者判定 (実装済)
- `src/win_panel.py::WinPanelDetector` — 「数値★WIN★数値」パネル存在検出 (閾値0.70)
- `scripts/collect_boards_lean.py` — `_SharedGameCounter` (行630-860付近)、`advance_if_new`、
  `observe_visual_active`、`_reconcile_boundary_anomalies` (行830-870付近)、
  CLI `--enable-boundary-multisignal` / `--enable-winner-panel-crosscheck` (行1999-2109付近)。
  W22 (早い者勝ち競合) は修正済 (`BOUNDARY_VISUAL_RISE_PERSIST_SEC`)
- `src/recognition_pipeline.py:3562-3739` — `is_match_active` 計算
  (`hard_match_off = score_zero_both or match_end_locked`、
  `effective_hard_off = hard_match_off and not score_actively_moving and not chain_in_progress`)。
  npzには `match_end_lockeds` 列が既に保存されている

これらは既定OFFで健全にテスト済。今回やるべきは (1) 新規盲点の診断とギャップ修正 (A)、
(2) 完全新規のSTABLE持続確認 (B) の2点のみ。

---

## A. 試合境界マルチシグナル: 新規盲点への対処

### A-0. 実フレーム確認済みの事実
`data/verify/diag_general_chain_contamination_2026-08-17/frames/030_c21_2P_f57548_f57548_d+0.png` は
**「やった!」(1P) / 「ばたんきゅー」(2P) の演出画面そのもの** — `MatchEndDetector` のテンプレ探索領域
(`SEARCH_P1`=(100,100,800,600) / `SEARCH_P2`=(1150,200,700,500)) にテキストが収まっている。
「新しい画面種別の追加」ではなく、**既存検出器がこの実例で不発だった理由の特定が先**。

### A-1. Step 0 (必須・最優先・診断のみ、実装しない)
`scripts/_diag_match_end_miss_2026-08-17.py` を新規作成し、video_c21 の該当区間 (f57548前後±3秒) で:
1. `MatchEndDetector.detect(frame)` を直接適用し `score`/`template_name`/`detected` をログ
   (NCC閾値0.55との差。W3型の単一ソーステンプレ精度劣化の可能性)
2. `match_end_locked`/`score_zero_both`/`chain_in_progress`/`score_actively_moving`/
   `hard_match_off`/`effective_hard_off`/`is_active` を同フレームで再計算し、
   どの分岐で `is_active=True` が生き残ったかを特定
3. 30秒チャンク再収集方式が `MatchEndDetector._last_detected_t` をチャンク境界でリセットし、
   演出開始を検出器が見る前にチャンクが終わっていないか確認 (W23と同型の測定人工物疑い)

**分岐 (診断結果次第でどちらか一方のみ実装)**:
- 診断A: テンプレNCC閾値未満 → (a) 複数動画からテンプレ追加採取+多数決 (W3教訓: 既存成功例への
  回帰チェック必須) or (b) 補助特徴 (低彩度・高コントラスト比率) とのAND合成
- 診断B: 検出済みだが他分岐 (`chain_in_progress`/`score_actively_moving`) が打ち消し
  → 該当ガードに match_end_locked 用の除外条件 (W22同型、工数小)
- 診断C: チャンク再起動の人工物 → 本番の通し視聴では非該当。148再収集が通し方式である前提を確認し、
  該当ならA-2スキップ可

### A-2. 実装 (Step 0の結果を踏まえてから)
フラグ `enable_result_screen_hardening` (既定False)。診断結果に応じ `MatchEndDetector` 内部
または `recognition_pipeline.py:3631` 近傍のガード条件のみを修正。
**新しい状態機械やシグナル種別は追加しない** — 既存の `hard_match_off` 経路をそのまま使う。

### A-3. 対戦カード紹介画面 (試合外5件のうち4件) への対処
`WinPanelDetector` は `is_match_active` に未配線 (grep確認済)。「パネル未検出→試合外」を
`hard_match_off` の3つ目の入力として追加する。
- フラグ `enable_win_panel_absence_gate` (既定False)
- 作用点: `recognition_pipeline.py:3631` 付近、`hard_match_off = ... or win_panel_absent_persist`
- 持続確認: W22の持続タイマーidiom流用 (瞬間的検出漏れで誤hard_offしない)
- **要検証**: 「非表示=対戦カード紹介」の弁別性は未実証。着手前に対戦カード紹介の実フレーム
  3枚以上で自己検収必須 (試合中演出にWIN★パネルが表示される実例は確認済み)

### A-4. 敗者の即時特定 (出力インターフェース)
`MatchWinnerDetector` は実装済・`enable_winner_panel_crosscheck` で配線済。**新規実装不要**。
c109/c13/c96 で通し、勝者判定一致率と unknown 率を計測するのみ。
出力は既存 `WinnerDetectionResult.winner` ("1P"/"2P"/None) を game_idx 単位メタデータに紐付け。

### A-5. 実装順序
1. Step 0診断 (最優先、他をブロック) → 2. A-2 → 3. A-3 (独立並行可) → 4. A-4 (いつでも並行可)

---

## B. STABLE確定の持続確認 (完全新規)

### B-1. 適用スコープ (トレードオフ結論)
**収集パイプライン限定**。RT表示・`board_state_machine.py` のSTABLE確定ロジックには一切触れない。
- 生ピクセル差分は「静止した非盤面画面」(ばたんきゅー等) を区別できない — それはAの守備範囲
- RT側で確定保留すると設置反映8フレーム基準・指摘13 (動くべき時に動かない) に抵触
- 収集側の「学習データ採用可否」列の追加なら、STABLE確定自体を変えず安全

### B-2. 実装
- 転用元: `scripts/_diag_general_chain_contamination_2026-08-17.py:160-176` の
  `_board_roi_gray`/`_column_diffs` → `src/board_motion.py` (新規、stateless) に昇格
- 挿入点: `recognition_pipeline.py::_step_side` のSTABLE履歴処理 (行7040以降、frame_bgr利用可)。
  side別ローリング差分バッファ (SEC基準の新定数 `STABLE_PERSISTENCE_CHECK_SEC`≈0.5秒)、
  確定直前N秒の列別diff最大値が閾値 (`STABLE_PERSISTENCE_DIFF_THRESHOLD`、実測ベース5.0起点) 未満なら
  `persistence_ok=True`
- 新規フィールド: `SideResult.stable_persistence_ok: bool = True` (backward compat)
- CLI: `--enable-stable-persistence-gate` (既定False)。**既定動作は列追加のみ・除外はしない** —
  除外は学習データビルダー側のオプション (W18教訓: 近似列の無自覚使用を防ぐため列の意味をdocstring明記)

### B-3. 反映遅延との整合
列追加+後段フィルタでありSTABLE確定タイミング・確定値を変えない → 8フレーム基準・指摘13への抵触は構造的にゼロ。

---

## C. 統合

### C-1. 依存関係
- AとBはコード接触点なし → 完全並行可能。両方ともW25コミット後に着手
- A-3はA-1/A-2の診断と無関係に独立着手可

### C-2. 検証方法
1. 一般分布35枚の汚染14枚を 無/A/B/A+B の4構成で再測定 — ③試合外5件のうち何件が識別可能になるか
   (期待値: Aで4-5件、Bは③に無効=0件。Bの主効果は①②の一部)
2. 物差しv2 (55盤面) 全域回帰、既定OFFでbit-identical
3. 品質ゲートFAILトリアージ3例 (c12白背景演出/c27ロビー/c57試合中盤) で副作用確認

### C-3. 148再収集をブロックする最小セット
**A-1診断 + A-2ギャップ修正のみ**。③試合外5件 (学習データだけを汚す最悪パターン) がAの守備範囲。
Bはブロッカーにしない (後続改善として並行)。

---

## 発注サマリ (コーダ向け)
1. [最優先] Step 0診断 (`_diag_match_end_miss_2026-08-17.py`、c21 f57548±3秒)
2. 診断結果に応じA-2実装 (`enable_result_screen_hardening`)
3. A-3実装 (`enable_win_panel_absence_gate`)、対戦カード紹介実フレーム3枚以上で自己検収後に提出
4. A-4実測 (新規実装なし、`enable_winner_panel_crosscheck` を c109/c13/c96 で走らせる)
5. B実装 (`src/board_motion.py` + `enable_stable_persistence_gate`)、35盤面+物差しv2で回帰確認
6. 全構成で一般分布35枚を再測定し③5件の解消率を報告、user viz提示

## 重要な留保
- Aの修正内容はStep 0診断まで確定できない。**診断スキップの決め打ちはW3/W19型の再犯 — 禁止**
- WinPanelDetector「非表示=試合外」前提は未実証 (A-3着手前に自己検収必須)
