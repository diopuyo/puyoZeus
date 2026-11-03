# A2: 段原点と開始受理の診断

対象: c13ea95 / R1s + D-v2h2b。事前登録は `DECISIONS.md` 末尾の「事前登録: A1・A2」。採否は統括。

## #6 の原因と観測根拠

入力は `wt_evalset/logs/eval_set/falsedeath_fix/collect/records/s1.jsonl.gz`。セット2第11試合（記録内 game=10）。
元診断は `review_2026-10-03/diag2/REPORT.md` と `case6_timeline.jsonl`。
`scripts/a2_inspect.py` が同じ入力から変化点を再抽出する。画像は使用・コミットしない。

| 元動画秒 | 1Pの観測 | 元式段/累積 | 意味 |
|---|---|---|---|
| 4259.666667 | formula_readで単発開始、表示得点389 | 1 / 100 | 前の単発。評価連鎖ID 5 |
| 4260.533333 | 実表示489 | 1 / 100 | 389→489、100点を独立確認 |
| 4261.433333 | STABLE、得点491 | 式累積100を保持 | 前連鎖の後に操作加点 |
| 4262.066667→4262.100000 | STABLE、実表示492→494 | 式累積100を保持 | 次の操作を観測。CHAIN中の段の区切りとは異なる |
| 4262.400000 | 新しいformula_read発火、表示494 | 2 / 200 | 新連鎖の初段へ前の1段・100点が混入。評価連鎖ID 6 |
| 4263.300000 | 実表示594 | 2 / 200 | 新連鎖の得点増分は100。200ではない |
| 4263.733333 | 次の式 | 3 / 520 | 局所段2、累積420。増分320 |
| 4270.533333 | 次の式 | 8 / 14380 | 局所段7、累積14280。増分5120 |
| 4271.866667 | 次の式 | 9 / 20780 | 局所段8、累積20680。増分6400 |

`FormulaStepAccumulator` は掛け算式の**読取りセッション**を数える。
`score_displayed=True` は幕間を通知するだけでセッションをリセットしない。
`FORMULA_SESSION_RESET_SEC=2.0` 未満の間隔で次の連鎖が出ると、前の単発を同じ式セッションに積む。
認識器の `_apply_chain_formula_early_fire` / `_apply_formula_step_update` はその累積段数を `chain_count` へ転記する。
評価器側では別連鎖IDとなる一方、式の段原点が前の連鎖を含んだままだった。
したがって、通知段をそのまま発火前盤面の予測段と照合するE35⁗′が1段先の予測増分を参照し、
prefire側も新連鎖の通知が段2なので `missing_first_formula` と判定していた。

保存記録は式の積・累積と通知を持ち、左辺・右辺の画素読取り値そのものは持たない。
単体テストでは同じ積100となる式を使い、同じ時間関係で式累積器の2段・200点を再現した。
これは画像の左右の数字を再判読したという主張ではない。

## 実装

新フラグ `--a2-safety-review` は既定OFF。本番採用リストは変更しない。
`FormulaOrigin` が、STABLEの確定盤面と数値得点表示を観測した間の操作加点
（正で連鎖の最小得点40点未満）を境界証拠とする。
その後の新発火で元式段が直前+1かつ累積増加なら、観測済みの前セッション段数・得点を原点にする。
元セッションがリセット済みなら補正せず、境界を消費して次段で誤補正しない。
欠けた段や不連続な通知から原点を推測しない。予測得点との一致探索はしない。

認識器・保存入力を変更せず、overlayへの入力を複製して補正する。
補正済み段数・式得点・式累積を、連鎖ID解決、prefire、途中段照合、得点帰属へ同時に渡す。
片側ごとに状態を持ち、試合境界で破棄する。frozen dataclassのライブ入力にも対応する。

## #6の効果と残る抑止

補正後はID 6の初段が1/100になり、直近の局所段6→7、7→8は
予測/実測とも5120点、6400点で一致する。段1→2以降の過去の段間増分も一致し、式と実表示の累積基準差は0。
4272.566667秒は「推定現在段9・観測段8・遅れ1」で `recent_consistent` となり、
E35の `post_counter_upper_bound` が1P死亡を証明。2P勝ち計算確定（1P=0.02）が回復した。
その後、4272.933333〜4272.966667秒の再判定は「推定現在段10・観測段8・遅れ2」で拒否する。
`STAGE_INTERVAL_SEC` と許容遅れ1段は事前登録外なので変更しない。
末尾の再判定だけを読むと、先に受理された証明を見落とすため、原票には両時点を残した。
prefireの `missing_first_formula` は解消したが、別の初段得点照合 `first_score` で拒否される。
prefire予測そのものが受理されたとは扱わない。

## H1′の開始条件と限界

各側について、既存の「現在STABLEかつ0〜2個」または
「境界後ずっとMENUで、STABLEを一度も観測していない」を要求する。
連続した試合番号、前試合の両側の非ゼロ得点、新試合の両側の最初の読みが0、
その0読みが境界後 `SEC_PER_HAND=0.733` 秒以内、という既存条件は維持する。
STABLEやCHAINを一度でも経由してからMENUへ戻ったケースは追加経路では受理しない。

テストは、成熟盤面上の0誤読、片側の非ゼロ初読、前試合得点の欠測、試合番号の飛び・同一番号、
期限超過、STABLE→MENU、CHAIN→MENU、初回観測だけのMENUを拒否することを確認する。
ただし、誤境界と両側の初読0とMENUが同時に与えられると、保存入力だけでは本当の開始と識別できない。
今回の追加条件も、この入力情報の限界を解消するものではない。実記録での過大方向は18記録の誤確定を別途採点する。

## 再現・検証

- `scripts/run_a2_batch.sh prod ...` と `a2 ...` が同じ再収集18記録を再生する。
- `scripts/a2_score.py` は既存falsedeath採点器を再利用。固定勝敗、表示区間、門の115試合合算（bootstrap 10,000回・seed 0）を維持する。セット別参考区間も既存設定（セット1 seed 20261001、セット2 seed 0、各10,000回）を維持する。
- `scripts/run_a2_reference.sh` は `git archive c13ea95 src scripts` のソースを使い、既存5記録の本番を再生する。
- `scripts/a2_verify.py` はNPZ・JSONLのバイト一致、診断JSONの辞書一致（既存`compare`と同じ定義）と全18記録のnice=10、BLAS=1、開始時空きメモリ6GiB以上を確認する。診断JSONの件数辞書はキー順が実行ごとに変わるため、バイト一致の成否も別項目で残す。
- `logs/a2_safety/` に記録別の再生・資源・採点原票を保存。画像は生成しない。

新しい出力ディレクトリでの再現手順（WSL、共有venvと既存モデルが必要）:

```bash
cd /mnt/d/puyo_analyzer/wt_a2
mkdir -p logs/a2_safety/reference
git archive c13ea95 src scripts | tar -x -C logs/a2_safety/reference
records=(review q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi c1 c2 c3 c4 c5 c6 s0 s1 s2 s3 s4 s5 s6)
bash scripts/run_a2_batch.sh prod "${records[@]}"
bash scripts/run_a2_batch.sh a2 "${records[@]}"
bash scripts/run_a2_reference.sh
bash scripts/run_a2_safety.sh score -m scripts.a2_score
bash scripts/run_a2_safety.sh report -m scripts.a2_report
bash scripts/run_a2_safety.sh verification -m scripts.a2_verify
bash scripts/run_a2_safety.sh start_audit -m scripts.a2_start_audit
bash scripts/run_a2_safety.sh case6_final -m scripts.a2_case6
```

再開時は完了票がある記録だけをスキップする。未完了の既存出力へは上書きしない。
原票の入力・出力SHA-256は `VERIFICATION.json` に残す。
固定ラベル用の既存E19原票は、評価worktreeの `logs/eval_set/set2/regression/e19/`
を読み取り専用で参照する。ラベル規則・内容は既存採点器のまま。

最終の18記録は全て最終実装で再生済み。修正の入力上の影響が0だったmia・c5も再生し直した。
開始未観測は57→22/176、追加受理35件は全てMENU/MENUかつ実表示得点0/0。
全表・門v2・消失と遅れは [検収表](A2_SAFETY_REVIEW_20261003.md)。

資源制御は `run_limited`（同時数6、Linux空き6GiB以上、nice 10、BLAS 1）を維持。
途中でB2・封印セット準備が追加されたため、`b2_recollect_run.py`と
`sealed_eval_prepare.py` / `sealed_eval_common.py` / `sealed_eval_run.py` /
`sealed_eval_boundary_choice.py`を読み取り確認し、A1と同じ
`scripts/a2_resource_limits.sh`へ揃えた。B2の`--one`なし、封印セットの準備親は
待機・割当だけなので除外。B2の`--one`は収集workerが存在する待機中だけ除外し、
子の終了後に実行する監査は数える。収集worker・DL・境界検出の子は数え続ける。
封印収集の`--phase full`も子への割当だけなので除外し、配列比較を行う`pilot`は数える。
切替時に終了したのは、子の計算がないことを検査したA2の起動待ちだけ。計算中の子や他レーンは停止していない。
