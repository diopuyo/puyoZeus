# D-v2h2 実装・検証記録

作業ブランチ: `claude/eval-set-zenchi-set1-20261001`。起点: `696929b`。
事前登録の正本: `<home>/.codex/worktrees/exev/puyo_analyzer/docs/agent_coordination/DECISIONS.md`
末尾「2026-10-03 D-v2h 検収と事前登録: D-v2h2」。採否は統括。

## 固定した構成と条件

`mode=v2h2` は `mode=v2h` のフラグ列に `--death-integrity-v2h2` だけを追加する。
新フラグは既定OFF。既存モデル・入力18記録・採点区間・bootstrap・認識構成は変更しない。
本番登録ファイルは変更しない。画像はコミットしない。pushしない。

判定は事前登録どおり: ①表示区間の誤確定0、②reviewの3:00場面≤2766.0秒・q第14試合の誤確定0、
④開始未観測がD-v2dより少ない。③のM_timeはD-v2d比の合算点推定≤+0.002なら既存の例外判断を継承する。
旧条件（点推定≤0かつ消失件数増加なし）は参考として併記し、普通の非劣化合格と混同しない。

## 修正と物理量の根拠

1. H1′受理時に両側がSTABLE、confirmed_boardが存在し、各盤面の非空セル数≤2であることを要求。
   境界から各側の最初の0読みまでの上限は `indicators_v2.SEC_PER_HAND = 0.733秒`。
   既存の1手時間（実測中央値）を使い、1組の設置が進む前の開始証拠に限定する。
   減算の丸めで境界のちょうど0.733秒を拒否しないよう、時刻を `boundary + limit` と比較する。
2. E35⁗′の照合候補は `post_counter_bound.base_predictions[chain_id]` だけ。
   hidden / recovery / midchain の観測適合候補を照合へ入れず、元予測が無ければ未観測とする。
   元予測の直近連続2段は観測へ照合し、過去の得点差は監査に残す。stage_regression等は永続的に拒否。
3. 現在段推定は `1 + floor((stamp - trigger_sec) / STAGE_INTERVAL_SEC)`。
   発火を1段目とし、`exchange_prefire_stage_timeout.STAGE_INTERVAL_SEC = 1.17秒` を使用。
   既存の23動画418イベントの式（2.61 + 1.17×N）の段間成分であり、今回の採点で選んでいない。
   最新観測段から2段以上進んでいれば未観測。予測最終段で時計を止めない。
   新しい式通知が無くても時刻の進行で外側キャッシュを更新し、古い照合結果を再利用しない。

いずれの上限も最初の梯子実行前に固定した。成績による変更は行っていない。

`MatchHistoryIntegrity` の追加オプションと `PostCounterDeathBound` の `early_exit` 以外の
追加オプションはkeyword-only。呼出し側も名前付きにし、rtのretain_audit位置引数と混同させない。
このworktreeのlandingのresult配線は起点で既に存在した。時刻も渡す形へ拡張し、resultを削除した変異が
実updateを通るテストで落ちることを確認する。CLIは `--death-history-boundary-start` 単独指定を拒否する。

## 検証・再現

- `scripts/run_falsedeath_v2h2.sh`: nice 10、BLAS各1、空きメモリ6GiB以上、既存run_limited。
- `scripts/run_falsedeath_v2h2_ladder.sh`: c5、mia試合5/7/9/10、review場面を先に検証。
  梯子通過時だけ全18記録を再生し、`falsedeath_score_20261001 --mode v2h2` で採点する。
- `scripts/run_falsedeath_v2h2_checks.sh`: 周辺回帰と既定OFFの本番5記録再生。
- `scripts/run_falsedeath_v2h2_compat.sh`: 旧v2hの短区間3記録について表示・イベントのバイト一致。
- `scripts/falsedeath_v2h2_coverage.py`: H1′の受理瞬間の証拠と全18記録の履歴一致を確認。
- `scripts/falsedeath_v2h2_report.py`: D-v2d比・D-v2h比、消失・回復の全例、遅れ分布、開始未観測の全表。
- `scripts/falsedeath_v2h2_receipt.py`: 入力ハッシュ、資源、静的検査、互換性、圧縮原票、所要時間。

ライブと評価の一致は共有overlayへ可変の再生入力とfrozenなライブ型入力を渡して検査する。
別ブランチの実ライブ統合・本番登録はこのコミットの対象ではなく、統括の統合レーンで確認する。

## 結果の要点

18記録の①②④は通過。③合算M_timeのD-v2d比は+0.001599641で+0.002以内。
ただし旧条件の点推定≤0は不通過。95%区間は[+0.000054594, +0.004047451]であり、
改善や通常の非劣化を示す結果ではない。D-v2h比も+0.001183101 [+0.000323076, +0.002310336]。

正しい確定の消失はD-v2d 71件→70件（回復2、新規消失1）、D-v2h 58件→70件（回復0、新規消失12）。
開始未観測は176試合中、D-v2d 65→57、D-v2h 22→57。安全側への厳格化に伴う検出範囲の後退を明記する。
3:00場面は2762.55秒。表示区間の誤確定・q第14試合の誤確定は0。

全表と監査原票は `logs/eval_set/falsedeath_fix/v2h2/REPORT.md`、`COMPARISON.json`、`RECEIPT.json`。
H1′の境界証拠は各フレームで盤面条件を再計算するため、最後のaccepted欄は受理履歴ではない。
受理件数には `START_ACCEPTANCE.json` の受理瞬間を使う（元の全体履歴との完全一致を確認）。

最初のセッションの中断後もdetachした梯子・全体再生は完了した。中断した回帰テストは
既存ログを残し、`regression_tests_resumed.log` と別の一時出力先で再実行。
全体採点後の実行コード差分はコンストラクタのkeyword-only位置調整と、50行上限を守るための
初期化ヘルパーへの移動（依存検査とフラグ代入）。評価計算は変更せず、最終コードのreview短区間を
既存v2h2原票とバイト比較する。本番5再生は通過済み、関数長の失敗は残りの周辺回帰を再実行して確認する。
所要時間は中断を含む経過時間、梯子・全体ワーカーの実時間、再生計算時間を区別して報告する。
