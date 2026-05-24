# セッション引継ぎ 2026-05-11 cycle 71b

## 現状サマリー (= cycle 70 → 71b への進化)

### 主要 課題と対応
1. **仮説 A** (= 置き誤認 → freeze) / **仮説 B** (= 連鎖検出 1 秒遅延) が判明
2. **Phase 1a 完了**: 物理推論主軸化 (= placement_inferrer + 連鎖即時判定)
3. **cell 認識 CNN メイン化**: cnn_override_prob 0.90 → 0.70
4. **cycle 71b 完了**: 案 A (連鎖整合性) + 案 B (縦/横幾何) で候補絞り込み強化
5. **ユーザーレビュー**: Phase 1a viz で「物理推論効いてる、 ただし置いた後の位置誤り + 修正速度」 が残課題

### ユーザー要件 (= 設計議論で明確化)
- 物理推論を「補助」 ではなく「メイン」 にする
- 落下中ツモは追跡しない (= 着地後だけ判定)
- 認識率 100% を目指す
- Phase 1a → 1b → 1c の段階分け

## 実装したファイル (cycle 71 〜 71b)

### 新規
- `src/placement_inferrer.py` (主要): 物理パターン全列挙 + NEXT 色固定 + 案 A/B 統合
  - `enumerate_landing_patterns`: 縦/横の物理妥当パターン列挙
  - `_classify_diff_orientation` (案 B): CNN 差分 2 cell の幾何で縦/横確定
  - `_chain_count_of` (案 A): ChainSimulator 整合性チェック
  - `infer_placement`: 主 API (chain_sim + score_delta 引数追加)
  - `resolve_after_placement`: 着地後即時連鎖判定
- `tests/test_placement_inferrer.py`: 20 件 pass
- `tests/test_hybrid_classifier.py`: 7 件 pass
- `scripts/diagnose_chain_transitions.py`: per-frame log
- `scripts/analyze_chain_diag.py`: 仮説 A/B 判別 + A=preempt 新カテゴリ

### 改修
- `src/hybrid_classifier.py`: DEFAULT_CNN_OVERRIDE_PROB 0.75 → 0.70
- `src/recognition_pipeline.py`:
  - `__init__` に `_chain_sim` 事前構築
  - `load_default` に `vote_mode` + `cnn_override_prob` 引数
  - `_build_hybrid_reader` に `vote_mode` + `cnn_override_prob` 引数
  - `_step_side` 改修:
    - 旧 `_compute_landing_inferred` 削除 (200+ 行)
    - 旧 `_inject_pseudo_chain_event` 削除 (40+ 行)
    - TSUMO_FALL→STABLE で `infer_placement` + `resolve_after_placement` 呼出
    - STABLE→STABLE でも `infer_placement` 適用 (= 取りこぼし補強)
- `src/image_reader.py`: ColorClassifier に `vote_mode` 引数 (= 投票方式、 default False)
- `scripts/visualize_recognition.py`: `--vote-mode` + `--cnn-override-prob` 引数

## テスト状況
- 関連 102/102 pass
- 広範 1149/1150 pass (= 残 1 件は flaky 合成テスト、 cycle 71 と無関係)

## 数値結果 (= v91_match1_75s_720p、 新評価軸 A=preempt 込み)

| 仮説 A | baseline (cycle 70) | vote | CNN-0.70 のみ | **Phase 1a** | **cycle 71b** (= 案 A+B 案) |
|---|---|---|---|---|---|
| A=preempt (= 物理推論先回り完了) | 136 | 166 | 122 | 144 | (走行中) |
| A=clean | 36 | 25 | 40 | 19 | (走行中) |
| A=hit (= 真の置き誤認) | 17 | 17 | 16 | 17 | (走行中) |
| A=partial | 0 | 3 | 0 | 0 | (走行中) |

**A=hit 17 件**が全版で安定 = 真の残課題、 cycle 71b で減ることを期待。

## 走行中ジョブ (= 再起動後も完走)

### v91 cycle 71b 評価 (setsid -f で detach 済)
- viz: `data/test_unknown/v91_match1_75s_viz_phase1b.mp4`
- diag: `data/diagnostics/v91_match1_75s_diag_phase1b.jsonl`
- log: `logs/viz_v91_match1_phase1b.log`, `logs/diag_v91_match1_phase1b.log`
- 完成見込み: 5-6 分 (= 17:35 開始、 17:40-17:42 完成見込み)

### 走行確認方法
```bash
wsl -d Ubuntu -- bash -c 'pgrep -af "python.*scripts" 2>&1'
```
プロセスがいなくなれば完走。 ファイルサイズ・行数で進捗判断:
```bash
ls -la data/test_unknown/v91_match1_75s_viz_phase1b.mp4
wc -l data/diagnostics/v91_match1_75s_diag_phase1b.jsonl
```

## 既に取得済 viz (= ユーザーレビュー済)
- Phase 1a viz: `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v91_match1_75s_viz_phase1a.mp4`
  - 観察: 物理推論効いている (= 全体品質向上)、 ただし**置いた後の位置誤り + 後から修正が目立つ**

## 次セッション優先タスク

### 即着手
1. **cycle 71b 完走確認** → diag summary 生成 → AB 比較表
   ```bash
   PYTHONPATH=. ./venv/bin/python -m scripts.analyze_chain_diag \
       --input data/diagnostics/v91_match1_75s_diag_phase1b.jsonl \
       --output data/diagnostics/v91_match1_75s_diag_phase1b_summary.md
   ```
2. **cycle 71b viz をユーザーにレビュー依頼** (= 案 A/B の効果目視)
3. A=hit 17 件が cycle 71b で減ったか確認

### 次の方向 (= AB 結果次第)
- **案 C 実装** (= 修正速度向上): STABLE→STABLE 経路で連続 N frame の認識ずれを検出し即時修正
- **Phase 1b 着手**: 連鎖開始 = scoreエリアの掛け算式 OCR、 終了 = 12 段目 col=2 出現
  - 仕様 memory: `reference_chain_phase_detection_spec.md`
- **学習データ作成** (= Phase 2 新規 CNN): 着地直後 cell 特化 CNN の学習データ自動収集
  - 自動ラベル元 = NEXT 履歴 (100% acc 確定)
- v89/v40 など他動画での regression 確認

## 注意点 (= 再起動後)

### self-hit による永久ループ注意
`pgrep -f visualize_recognition` のような pattern は **bash 自身の cmdline も hit** する → 永久ループ。 対策:
```bash
# 悪い (= 自己 hit する):
pgrep -f visualize_recognition
# 良い (= python プロセスだけ hit):
pgrep -f "python.*visualize_recognition"
```
ただし上記でも bash の cmdline に "python.*visualize_recognition" がそのまま含まれているとまた hit。 完全回避は:
- `pgrep -fa cmd | grep -v "bash -c"` でフィルタ
- もしくは monitor / sleep のシンプル方式

### シェル escape 注意
wsl 経由で `$(...)` を渡す時、 Windows bash の処理で escape が壊れる場合あり (= `\$` が `1` に化けた事故あり)。 single quote で wsl コマンド全体を囲む方が安全。

### memory 参照
- `feedback_autonomous_operation.md`: 自律運転前提
- `feedback_msys_pipe_escape.md`: パイプ・特殊文字注意
- `feedback_chain_phase_physics_only.md`: STABLE 以外 CNN 信用しない
- `reference_chain_phase_detection_spec.md`: **新規追加** Phase 1b 仕様 (= 連鎖開始/終了の正確検出)

## ファイル位置サマリー
| カテゴリ | パス |
|---|---|
| 実装 主軸 | `src/placement_inferrer.py`, `src/recognition_pipeline.py` |
| 既存テスト | `tests/test_placement_inferrer.py`, `tests/test_hybrid_classifier.py`, `tests/test_image_reader.py` |
| 診断スクリプト | `scripts/diagnose_chain_transitions.py`, `scripts/analyze_chain_diag.py` |
| viz 出力 | `data/test_unknown/v91_match1_75s_viz_phase1a.mp4`, `_phase1b.mp4` (走行中) |
| diag JSONL | `data/diagnostics/v91_match1_75s_diag_*.jsonl` |
| diag summary | `data/diagnostics/v91_match1_75s_diag_*_summary*.md` |
| 引継ぎ doc | `docs/HANDOFF_2026-05-11.md` (cycle 70), `docs/HANDOFF_2026-05-11_cycle71b.md` (本 doc) |

## 次セッションキックオフプロンプト
別途同階層に `KICKOFF_2026-05-11_cycle71b.md` として保存。
