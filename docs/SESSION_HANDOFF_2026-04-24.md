# セッション引継ぎドキュメント — 2026-04-24

前セッション (2026-04-23 午前〜2026-04-24 02:00頃) で実施・決定した内容のスナップショット。

## 1. 到達した成果

### 1.1 CNN パッチ分類器の汎化性能
| 指標 | 値 | 備考 |
|---|---|---|
| internal best | **99.37%** | Cycle 1 |
| **holdout best** | **89.20%** | **Cycle 3 (peak)** |
| CNN アーキテクチャ | residual (24ch + ResBlock) | Phase 2 で優勝 |

7 サイクル自律学習で Cycle 3 が peak、以降下降 → 4 サイクル停滞で **early_exit 自動発動**（13:47）。

### 1.2 ホールドアウト回復ストーリー
旧パイプライン 85-88% → OOM クラッシュで新規 DL データ混入 → 新パイプライン Cycle 1 で 72.82% に急落 → 緩和策の効果で Cycle 3 に 89.20% まで +16pt の大幅改善を達成（peak はそこで頭打ち）。

### 1.3 適用済み緩和策（`scripts/long_improve_v2.py`）
- `oversample` 最大 8倍 → **3倍**（ハード例過適合抑制）
- Focal gamma 2.0 → **1.5**
- Adam に `weight_decay=1e-4` 追加
- Phase 1 `patience` 3 → **2**
- `balance_dataset(empty_ratio_cap)` 0.35 → **0.40**
- **stagnation_limit=4** で意味のない処理継続防止

## 2. システムアーキテクチャ（自動復旧完全版）

### 2.1 訓練パイプライン
- `scripts/long_improve_v2.py` 本体（40時間目安モード）
- `scripts/run_long_improve_wrapper.sh` で Python クラッシュ時再起動（最大30回）
- wrapper 内にハートビート監視サブプロセスを仕込み watchdog を相互監視

### 2.2 Windows 側監視
- `scripts/watchdog.ps1` (PID 17004 稼働中) が 120秒毎に:
  - WSL VM 停止 → `wsl -e true` で起動
  - 学習不在 → wrapper.sh を `setsid nohup` で spawn（ただし `data/training_stopped` マーカーがあれば skip）
  - Claude Code 不在 → `claude -c` で最新セッション再開
- `data/watchdog_heartbeat` を 120秒毎に touch
- HKCU レジストリ Run キー `PuyoAnalyzerWatchdog` 登録済 → Windows ログイン時 `watchdog_bootstrap.ps1` が自動実行

### 2.3 milestone ロギング
`scripts/long_improve_v2.py` が `data/milestones.jsonl` に重大イベントを追記:
- `cycle_start` / `cycle_complete` / `new_best` (+0.5pt以上) / `threshold_99` / `e2e_result` / `discovery` (99%かつholdout≥95%) / `anomaly` / `phase_complete` / `early_exit` / `fatal`

`scripts/poll_milestones.py` で未読分を標準出力に出し、`data/milestone_last_seen` を更新。

### 2.4 Claude Code 側監視ループ
`/loop` 動的ペース:
- 通常 20 分 / cycle_complete 検出で 270秒 / fatal で 180秒 / idle 2連続で 30 分延長
- （現状は停止中。再開は新セッションで `/loop` 再呼び出し）

## 3. 修正したバグ

### 3.1 load_all_data() OOM（解決済）
322 万パッチを `np.concatenate` → 18GB 要求 → WSL2(16GB) OOM クラッシュ。ファイル単位で比例サブサンプリングしてから積む方式に変更。

### 3.2 `src` モジュール import 失敗（解決済）
scripts/ から Python 起動時に `from src.board import ...` 失敗。`sys.path` にプロジェクトルートを追加。wrapper.sh でも `PYTHONPATH` 設定。

### 3.3 Board.grid 属性エラー（解決済）
`evaluate_indicators()` で `b1.grid` を参照していたが Board には `grid` 属性なし。`b1.count_puyos()` に修正。

### 3.4 2P 盤面読取り全セル 0 問題（解決済）
評価入力にモンタージュ PNG（848×1630）を使っていたが `read_both_boards` が 1920×1080 にリサイズ → 2P region が黒背景を指していた。`data/frames/sample/` の 1920×1080 生フレームを使うよう変更。

## 4. 指標式の大幅修正（別エージェント実施）

`src/indicators.py` と `src/scorer.py` の多数の欠陥を修正:
- **副砲の質**: 窒息盤面で即 0 返し、viable 条件を `chain≥2 AND erased≥8` に厳格化
- **相殺力**: ぷよぷよ公式得点式（連鎖パワー＋連結＋色数ボーナス）実装、`MAX_OJAMA_OFFSET=72`
- **窒息リスク**: 非線形マップ `1-exp(-(h-9)²/8)` を線形成分とブレンド
- **本線完成度**: `MAX_EXPECTED_CHAIN=10`
- **催促耐性**: base_chain=0 時は中立値 0.5
- **フィールド効率**: 未発火時は「連結≥2 のぷよ数/全ぷよ数」
- **Scorer** `EVEN_THRESHOLD=3.0`

`tests/test_indicators.py` + `tests/test_scorer.py`: **369 passed / 1 skipped** リグレッションなし。

## 5. 未解決・次アクション候補

### 優先度 高
- **Cycle 3 の best model 復元**: `models/cnn_best.pt` は Cycle 7 時点で上書きされている可能性。復元するなら `cnn_p1_r01.pt` (Cycle 3 保存) を確認。
- **Cycle 7 sanity=False の調査**: 最終モデルで初発生。`scripts/e2e_validate.py` を読み violation の具体内容を確認。
- **指標実データ検証**: 指標修正の成果を CNN（holdout 89%）と組み合わせて実盤面で確認。

### 優先度 中
- watchdog の WSL wake スキップ最適化（training_stopped 時は WSL も起こさない）
- 相殺力の対数スケール化（早期飽和対策、別エージェント提案）
- Scorer の非線形変換（窒息・相殺の閾値型マッピング）

### 優先度 低
- 連結ボーナステーブル 11+ 連鎖拡張
- 伸ばし余地の 2ツモ先読み
- 指標間相関除去（本線完成度×相殺力の重み再校正）

## 6. 環境

- 稼働 PID 一覧（現状）: watchdog 17004 / wrapper DEAD / python DEAD
- 停止マーカー: `data/training_stopped` (touch 済、watchdog が尊重)
- 学習再開: `wsl bash -c 'rm /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/training_stopped'` で可能（ただし holdout は頭打ち）
- 権限設定: `.claude/settings.local.json` に `defaultMode: bypassPermissions` + 広めの allowlist 設定済
