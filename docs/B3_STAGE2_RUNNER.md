# 段2実行器

正本は `B3_PREREG_2026-10-05.md` の段2・追記A〜D。封印セット用の実行器ではない。
本物のB3モデルはこの実装・試走では作成も読込もしていない。

WSLで作業ディレクトリを `D:/puyo_analyzer/wt_b3` にして、次を実行する（出力先は新規ディレクトリが必須）。

```bash
PYTHONPATH=. <home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m scripts.b3_stage2_run \
  --dry-run --rows /mnt/d/puyo_analyzer/b3_data/eval_rows_v2b/b3_zenchi_merged.parquet \
  --out /mnt/d/puyo_analyzer/b3_data/stage2_dryrun_new \
  --exev-root <home>/.codex/worktrees/exev/puyo_analyzer \
  --reference-root /mnt/d/puyo_analyzer/wt_a1/logs/a1b/combo \
  --labels-root /mnt/d/puyo_analyzer/wt_evalset/logs/eval_set \
  --calibration /mnt/d/puyo_analyzer/wt_a1/data/indicators_v2/winprob_calib.json --review-csv
```

最終モデル完成後は `--dry-run` を `--model /path/to/final.joblib` に置き換える。
段1実行器は `--run --save-final /path/to/final.joblib --final-config small|ref|large --final-seed 1003`
で、明示された設定を全学習行にfitして保存できる（通常の `--rows-dir --targets --out` も必要）。
設定の選択・本物の学習実行は今回行っていない。CV・部分採点と最終保存は別実行。
保存・読込APIは `src.b3_train_model.save_model/load_model`。特徴順とNaN除外を含むTwoStateModelのjoblibで、既存ファイルを上書きしない。

## Rと採点の契約

- F12から `wt_evalset/logs/eval_set` の固定ラベルへ、採用記録・`a1b_replay.py` から `wt_a1/logs/a1b/combo/replay/a1/*` へたどる。互換キー`a1`の実フラグはA1b+A2。`COMMAND.json`と完了票を検査する。
- Rの`RESULT.json`は旧本番との比較を含むが、今回は **a1側だけ** がR。旧`prod`側をRとして使わない。
- F2のSHAと既存M_time採点器・誤確定走査・平滑化ソースを照合。入力原票・ラベル・モデル・行・較正のSHAを保存する。
- Rも候補も既存`eval_set_score_20261001.score`で再採点する。Rの各パートの30Hz時刻をそのまま使う。115試合を減らさず、最初の行より前・2秒超の欠測を0.5で課点。試合間の保持は禁止。
- **既存原票は`display_p1`が平滑前確率、`display_adv`が平滑後adv**。候補もこの定義を保つ。EMAは本番クラス（0.25、ブレンド0.5秒）、確率と非線形変換後advを別々に平滑化。発火・終了・着地等のR観測時刻を切替の合図として使う。
- B3の`p_static/p_exchange/p_selected`は死亡証明を入れる前のモデル出力。非採用側は欠測。LL/Brierはこの`p_selected`、Rは原票の平滑前`display_p1`で計算する。R原票は死亡証明由来の確率も含む。この定義を受領票にも記載する。
- M_agreeは終盤の的中フレーム総数/終盤フレーム総数で、試合別比率の単純平均へ変えない。セット別・115試合全体とも対応付き試合bootstrap 2,000回、seed1003、百分位95%区間。
- F10の独立証明はRと完全同じフレーム・確率・sourceを使う。イベントも確定証明だけを転写して既存走査器で再走査。消失・遅れは試合/側で対応し、遅れ0もp95に含む。候補だけの確定は別原票に残す。
- 短5記録・第14・3:00の安全回帰は同一証明のR採点原票を継承。追加4記録の証明フレームと入力SHAも`inherited_safety/`に保存する。18記録・短5・115は重複集合で独立標本として足さない。
- セット2第1試合は行内`won_p1`が欠測。採点は全試合とも既存固定ラベルを正本とし、行内に値があれば一致を検査する。

## 出力・未実施事項

`RECEIPT.json`に母数・根拠・機械判定、`scores.json`に試合別値・R対差・確定監査、`replay/*`に既存採点器用原票を保存。
`predictions_set1/2.parquet`は原向きだけの行と予測。元動画時刻は`t_source_sec`。
`src.b3_stage2_trace.review_rows(rows, predictions)`が行キーで照合し、`from_b3_rows_iter`へ接続する公開口。
`--review-csv`でCSVも保存する。Rの平滑後確率は原票に保存されていないため`p_ref_display`は欠測のまま。
レビュー7窓とOBS相当30分の正式時間門は別作業で、未実施をPASSへ置換しない。forecast72はF10で対象外。
仮モデルの受領票は「検証用の仮モデル。数値・合否に意味はない」と明記する。
