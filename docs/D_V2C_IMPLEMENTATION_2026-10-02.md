# D-v2c 実装・実行

事前登録の正本は exev の `docs/agent_coordination/DECISIONS.md` 末尾
「2026-10-02 事前登録: D-v2c」。構成は D1 + D3′ + D4 + E35″。
採否は統括。本番設定は変更しない。

## E35″ の観測

新規フラグ `--post-counter-current-puyo-integrity-v2c` は既定OFF。
既存 `--post-counter-current-puyo-integrity` は常に未知ツモを前置する従来動作を維持する。

未知ツモを省く条件は、受け側の実状態が CHAIN で、既存の絶対終了判定器の
`_abs_saw_chain` と `_abs_baseline_ready` が成立していること。
後者は既存の「実CHAIN観測後、開始設置の移動が収まったNEXTを基準化する」観測であり、
新しい設置推定・閾値・待ち時間は追加しない。
現在の NEXT1 がその基準、NEXT2 が既存 `_queue_successors` の観測値に一致し、
現在の NEXT スライドがなく、終了ラッチも `end_signal_sec` も未成立のときだけ省く。
観測不足・基準未確定・他の状態・移動/終了観測があれば、従来 E35′ と同じ未知ツモを前置する。

状態・基準・開始設置確定・NEXT1/NEXT2・移動・終了・発火時刻を
`current_puyo_observation` として events / post_counter_bound_audit に保存する。
未知ツモ前置は `unknown_current_pair_prepended`、自連鎖中の省略は
`unknown_current_pair_omitted_self_chain`。観測の変化は評価キャッシュにも反映する。

## 査読指摘

- C2: D3′有効時のみ、events の各 chain に `independent_score_required` と
  `finalized_evidence_score` を保存。E35′/E35″有効時のみ、証明の前置有無を保存する。
  OFF原票のスキーマ・バイト列を変えない。
- C3: 本番フラグ適用後のCLIと、直接呼出しのoverlayで E35 本体 OFF の v2系指定を拒否する。
- C6: 本番原票5記録（review/q/fc/mia/zenchi）を再生し、display.npz と events.jsonl の
  ファイルバイトを比較する。5記録のテストを除外しない。
- 正の対照: 本当に死亡するおじゃま床の局面で、未知ツモを前置した E35′/E35″の両方が
  dead を維持することを検査する。

## 固定実行手順

`scripts/run_falsedeath_v2c_checks.sh` は共通回帰とC6の5記録、
`scripts/run_falsedeath_v2c_ladder.sh` は短区間→条件成立時のみ18記録→採点。
すべて `scripts/run_falsedeath_v2c.sh` → 既存 `run_limited` 経由、nice 10、BLAS 1、
開始時空きメモリ6GiB以上。既存の計算本数上限を変えない。
待機専用であることをコード確認した `plan_c_t_controls`、`_diag_gfe_stage2_capture_all`、
`_diag_gfe_stage2_finish` を、従来の制御器2本に加えて計算本数から除外した。
いずれも実計算は独立した子モジュールで実行し、その子は上限6本に数える。
既存の共通 `run_limited` 本体・他worktreeは変更していない。
WSL内の実行ディレクトリは `/mnt/d/puyo_analyzer/wt_evalset` に固定する。

短区間は既存の診断4事例と review 3:00場面。
4件の誤確定0かつ3:00場面の既存指標が2766.0秒以下でなければ停止する。
全体は `mode=v2c`。旧条件・門v2・warmup・q第14試合・本番比の消失/遅れを維持する。
省略回数はキャッシュ再利用も含む実証明呼出し単位で数える。
後続死亡は同一記録・同一試合・同一側の当該時刻より後の観測死亡だけと照合する。
未観測は生存とは断言しない。全表・原票は `logs/eval_set/falsedeath_fix/v2c/`。
