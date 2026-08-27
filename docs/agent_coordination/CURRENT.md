# 現在地

## 2026-08-27 23:25 JST — userレビュー修正 v35・5試合動画完成

更新者: Codex

- 2:10は、相手連鎖終了後の盤面と自分が残り時間で組む盤面の予測が必要なため、
  user合意どおり今回は先送り。v30/v35の source=216.467 は
  **adv/P1勝率ともbit-identical（P1 71.00% / P2 29.00%）**。
- 3:31のP1 10%は、実際の5連鎖対応を開始時のChainEventが
  **1連鎖・40点**、相手を2連鎖・1,960点と過小評価し、予測着弾28個を
  決着値として固定したことが根因。進行中の片側だけが40点下限、相手は40点超、
  かつ直前STABLE評価と逆方向へ未解決90/10上限を超える場合だけ、直前評価へ
  保留する既定OFFガードを追加した。source=297.433はP1 **10%→54.15%**、
  実量が揃うsource=301.700はepisode台帳を優先しP1 **72.36%**。
- ガード範囲を全行比較で絞り込み、v30比7,306行中の差分は
  **source=297.433〜301.667の128行だけ**。双方小連鎖や同方向の確信度変化は
  従来評価を維持する。手作り重み・連鎖数の上乗せは追加していない。
- 3:51は同じ交換内の再決着でside-local追跡が消える不具合と、resolver仮想盤面
  対実盤面という異系列差分を修正。観測した同一系列の単発前後差は
  model -79.696→-73.569（P1方向 **+6.128**）、hold +42.240→+48.368。
  未解決上限により表示はP1 **89.22%→90.00%**。適用1回、二重適用0。
- 先頭5試合v35実測: 9,755表示行、1秒150点急変 **0**（最大87.839）、
  未解決上限違反0、active chainをRESOLVED表示0、違法hard適用0、
  global保存則0/9,755行、gross保存則0/4,302判定行。映像+音声の全編デコードPASS。
- テスト: 変更関連 **268 passed**。全pytestは今回完走させていないため主張しない。
- レビュー動画:
  `data/verify/gate4_first5_review_2026-08-27/
  gate4_first5_cond5_v35_review_extreme_flip_guard.mp4`。
  旧v29〜v34成果物は無変更、`src/production_config.py`未変更、
  本番ON・production登録なし。

## 2026-08-27 20:17 JST — userレビュー修正 v29・物理chain_idゲート完成

更新者: Codex

- user指摘3:31の誤判定はv18の構造修正を維持し、実動画 source=301.700 で
  **P1 72.36%**（旧1%）を確認。交換中にresolverの物理連鎖を優先し、古い
  resolved表示へ戻す経路を閉じた。
- user指摘3:50に対し、hold開始後の新規side-local連鎖を物理chain_idで識別。
  同一本線の継続を誤って新連鎖とした source=302.833 はroot chain_id一致で拒否し、
  真の1連鎖 source=316.500だけを開始1回として追跡した。直結回帰を新設。
- 真の1連鎖後のfresh盤面を全文脈で評価すると、anchor +42.240に対しモデル差分
  **-70.945**、候補 -28.705（P1有利→2P有利へ反転）だったため、既定の方向安全弁で
  拒否し **P1 89.22%を維持**。第三者設計レビューも「必ず動かす補正は不可」と判断。
  現行`second_chain_potential`は期待セカンド火力でなく「本線非参加色ぷよの割合」なので、
  この場面の根治は次の指標壁打ちで再設計する。手作り重みは追加していない。
- 2:10のP2 29%はuser指定どおり低優先度の残課題として今回は不変。最終試合の
  1秒150点急変は0を維持し、表示は未解決90/10上限内。
- 先頭5試合v29実測: 9,755表示行、1秒150点急変 **0**（最大87.839）、
  未解決上限違反0、active chainをRESOLVED表示0、違法hard適用0、
  global保存則0/9,755行、gross保存則0/19,510 side、close済み保存則0/6、
  異常累積0/7。映像+音声の全編デコードPASS。
- 正式Gate用verifierは違反0だが、レビュー切出しを第5試合直後411.633秒で止めるため
  segment-end OPEN=1/1（未行使1）としてFAIL。会計破綻ではなく切出し終端条件。
- テスト: 変更直結259 passed。全testsは94%までfailed 0（既定skipのみ）だったが、
  `/mnt/c/Windows/Fonts/meiryo.ttc` のWSL Plan9 I/O待ちが長時間化したため中断。
  全pytest完走の主張はしない。
- レビュー動画:
  `data/verify/gate4_first5_review_2026-08-27/
  gate4_first5_cond5_v29_review_physical_chain_id_guard.mp4`。
  旧成果物は無変更、`src/production_config.py`未変更、本番ON・production登録なし。

## 2026-08-27 15:36 JST — 条件5 v16・先頭5試合レビュー動画完成

更新者: Codex

- Claude全体レビューの指摘を独立再現。v14では
  `hard_override_target` と `net_raw` の方向不一致が462/1,050 frameあり、
  物理ターゲットの方向へ表示が振られていた。v15で確定量方向と交換全体方向の
  一致を必須化し、方向不一致を0へ是正した。
- 追加の根因を実測確定。未解決中の±100禁止を評価値±90へ丸めていたが、現行較正では
  ±90=勝率98.9%/1.1%であり、実質的に99%↔1%急変を残していた。v15の急変8件中
  7件にこの経路が関与。v16はepisode専用上限を勝率90%/10%相当
  (`adv≈±43.92`)へ変更し、既存B案の`KILL_UNCONFIRMED_ABS_CAP=90`は互換維持した。
- 未解決台帳と逆向きの旧live/hold致死候補を361 frame拒否。前試合のsticky死亡確定は
  監査列へ残しつつ、次試合の`PhysicalContext`から除外。EVEN帯を方向投票に使わず、
  episode gateでadvだけ変わったときの勝率同期と、EVEN外のadv/勝率表示矛盾も是正した。
- 先頭5試合の実測: 1秒150点急変 **8→0**、target/net逆方向 **462→0**、
  40秒局面は1P有利24 / 1P **76.8%**を維持。gross保存則0/19,510 side、
  global保存則0/9,755行、closed保存則0/6、違法hard適用0、会計verifier PASS。
- レビュー動画:
  `data/verify/gate4_first5_review_2026-08-27/
  gate4_first5_cond5_v16_review_probability_bounded.mp4`。映像+音声の全編デコードPASS、
  40秒フレームを目視確認済み。旧v14/v15証跡は無変更。
- テスト: 変更直結393 passed、関連範囲522 passed。全testsは95%までfailed 0だったが、
  WindowsフォントI/Oの単一テストが長時間化したため中断（全完走の主張はしない）。
  `src/production_config.py`未変更、本番ON・production登録なし。

## 2026-08-27 07:39 JST — 条件5v10先行NG / v11未確定方向±90

更新者: Codex

- v10先頭3区間で致死弱化2件。台帳方向は実勝者と一致していたが、未解決の
  ±100候補を止める際に方向まで生モデル値へ戻し、最終表示が-21.756/+1.509へ
  弱まっていた。
- 仕様は「未解決中も向きは出してよい、±100断定だけ禁止」。v11は既存定数
  `KILL_UNCONFIRMED_ABS_CAP=90`を再利用し、±100候補を方向付き±90へ丸める。
  生モデルが同方向に既に±90を超える場合は弱めない。
- 関連302テストPASS。v10は3区間完成時点で停止し証跡保持。
- v11固定snapshotを作成し、全8区間を3並列で07:39開始。終了見込み09:20前後。
  `src/production_config.py`未変更。本番ON・production登録なし。

## 2026-08-27 06:47 JST — 条件5v9先行NG / v10物理勝者の直接確定

更新者: Codex

- v9 seg01で元の急反転は抑えたが、実勝者1Pの最終表示が+100→+25.941となり
  致死弱化1件。t=221.867で死亡確定は正しく`hard_override_target=+100`を
  出していたが、ゲートは逆向き候補を止めるだけで確定値を表示へ採用していなかった。
- v10は決定不変性のtargetを「許可方向」ではなく物理確定値として適用する。
  未確定中は完全上書きを抑え、死亡確定後は勝者方向へ±100を直接入力する。
- 短窓実測で死亡確定後+21→+95.6→+99.97→+100へ収束。逆方向適用0、
  物理勝者方向訂正374/374、監査矛盾0/1,800行。関連301テストPASS。
- v9はseg01完成時点で停止し証跡保持。v10固定snapshotを作成し、全8区間を
  3並列で06:47開始。終了見込み08:25前後。
- `src/production_config.py`未変更。本番ON・production登録なし。

## 2026-08-27 06:11 JST — 条件5v8先行NG / v9台帳純残量配線

更新者: Codex

- v8の先頭2/8区間で致死弱化は0へ戻り、張り付き8.68%→4.07%、逆符号
  2.12%→1.28%、反転6→4試合、急変11→7へ改善。一方、決着逆方向は5→8。
- seg02 game8では台帳が`net_raw=-553`（2P攻撃優勢）を正しく出しているのに、
  致死判定は台帳を読まず通常予告だけで1P側+70を出していた。条件5は旧累積器と
  排他なのに、置換先の台帳純残量が致死入力へ未配線だった。
- v8残り6区間は時間最適化のため停止。完成2区間と部分logは削除せず保持。
- v9で`net_raw>0`を2P受け、`net_raw<0`を1P受けへcapなしで配線。相殺後の
  純残量なので両側へ二重加算しない。関連300テストPASS。
- v9固定snapshotを作成し、全8区間を3並列で06:11開始。終了見込み07:50前後。
  `src/production_config.py`未変更。本番ON・production登録なし。

## 2026-08-27 05:34 JST — 条件5v6会計PASS・表示NG / v8全区間再測定

更新者: Codex

- v6は全8区間完走。保存則はgross 0/422,014 side、global 0/211,008行、
  close済み後着同期0/88、未帰属0、重複0、違法hard override 0で会計層はPASS。
- 表示は張り付き6.42%→4.33%と改善した一方、急変27→30、決着逆方向23→25、
  真の致死弱化6件でNG。会計ではなく表示ゲートの実装契約違反を確定した。
- 仕様は「未解決中の±100完全上書きだけ禁止」だが、v6は部分補正まで全停止していた。
  また早期解除がbooleanだけで、確定した勝者と逆向きの±100も許せた。
- v8で完全上書きだけを対象化し、未解決中は物理的に確定した勝者方向と一致する
  ±100だけを許可する。方向をsidecarへ追加し、未解決/許可/方向の整合も全行検査する。
- 関連351テストPASS。v7短窓で新監査列の契約不整合を検出したためv7は証跡として保持し、
  解決済みepisodeの方向制約を消したv8短窓で矛盾0/1,800行、違法適用0を確認。
- v8固定snapshotで全8区間を3並列測定中。終了見込み07:15前後。
  合格後は全pytest、先頭5試合レビュー動画、続いて30先全体（109試合、
  86.467〜6972.933秒）を別名・別出力先へ生成する。
- `src/production_config.py`未変更。本番ON・production登録なし。

## 2026-08-27 01:18 JST — 条件5v5 NG / v6全区間再測定

更新者: Codex

- v5は7/8完成時点の先行検証でNG。gross/global保存則と違法hard overrideは0だが、
  後着同期3/55、未帰属2区間、重複1区間を検出した。seg08は部分logを残して停止。
- 後着同期・重複の根因は、close後も同一chainの`STEP`成長が続くのに、旧要約への
  backfill対象がFINALIZE/CANCEL/LANDだけだったこと。STEPが新episodeへ入り、
  FINALIZE時に旧要約が過去の全成長を一括反映していた。
- FIRE/STEPも旧要約へbackfillし、新episodeへtouchしない。遅延成長の件数・量と
  dropped件数・量を新設し、後着outstanding差とclosed要約差を直接検算する。
- 未帰属2区間は正式試合境界で未照合計上・chain退役した後、同frameの予告消失を
  settlementとして再適用した二重処理。境界frameのsettlementを除外し、境界数・
  除外件数・除外量を監査列へ出す。
- 新しい最小再現2本は修正前2 failed→修正後2 passed。関連152+resolver43=
  **195 tests passed**、py_compile成功。
- v6 snapshotを新設し、全8区間を3並列で01:17開始。v1〜v5は上書きせず保持。
  `src/production_config.py`未変更、本番ON・production登録なし。

## 2026-08-26 23:36 JST — 条件4正式NG / 条件5v5全区間再測定

更新者: Codex

- 条件4は8/8完了。条件1比で急変27→21、反転18試合/23回→15試合/20回へ
  改善したが、張り付き6.42%→9.32%、逆符号1.95%→2.58%へ悪化し、
  決着逆方向23/109は不変、真の致死弱化1件。必須0件を破るため**正式NG**。
- 条件5v4の2区間合算で、後着同期1/35と重複抑制1件を新たに検出。
  同期1件は同frameの旧episode決済(-1)と新chain生成(+2)を全体差で混ぜた検証器の
  誤検出。close済みchainだけの符号付きoutstanding差を新監査列にして分離した。
- 重複1件は実不具合。旧episodeへの後着イベントを同時にOPENな新episodeへも
  `touch`していた。backfill先があるイベントは旧要約だけへ帰属させるよう修正。
- 修正前の最小再現2本は2 failed、修正後2 passed。関連150テストとresolver 43テスト、
  合計193テストPASS。v4の完成2区間・部分3区間logは上書きせず保持。
- v5固定snapshotを新設し、全8区間を3並列で再測定中。最初の3本はseg01/02/03で、
  seg02は今回の重複再発、seg03はv3の累積点下振れ再発を直接検査する。
- `src/production_config.py`未変更。本番ON・production登録・レビュー動画生成なし。

## 2026-08-26 20:56 JST — 条件2正式NG / 条件5v4再測定

更新者: Codex

- 条件2は8/8完了。条件1比で張り付き6.42%→5.87%、逆符号1.95%→1.74%、
  反転18試合/23回→14試合/16回、急変27→22と改善したが、決着逆方向は23/109で
  不変、seg05 game2の真の致死弱化1件が残った。必須0件を破るため**条件2は正式NG**。
  条件2+5の追加測定は行わない。
- 条件5v3はseg01/02を完了したが、seg03のt≈2088秒以降で同一chainの暫定量が
  1→0へ下がり、追加専用台帳の不変条件が例外を出して停止。失敗logとv3成果物は保持。
- 根因は同一物理連鎖の累積得点OCRの一時下振れ。累積点は物理的に減らないため、
  `ChainIdResolver`でrunning maxを保持し、無視件数とFORMULA_STEP総数を別々に
  sidecarへ出すv4修正を実施。修正前に最小再現2本が失敗し、修正後は関連191テストPASS。
- v4固定snapshotは`data/verify/gate4_condition5_2026-08-26/
  _snapshot_cond5_codex_20260826_v4`。v1〜v3は上書きせず監査証跡として保持。
- 現在は条件4 seg03/04の2並列と条件5v4 seg01の計3本を実行中。
  `src/production_config.py`未変更、本番ON・production登録・Gate 4採用判定なし。

## 2026-08-26 18:36 JST — 条件5v2実データNG修理 / v3再測定

更新者: Codex

- 条件5v2の正式seg01で、gross保存則0/53,620 side、global保存則0/26,811行、
  違法hard override 0を確認した一方、後着更新同期1/5とnormal close後の一時未照合
  1/3を検出し、v2をNGとした。修正前seg01と部分ログは無変更保持。
- 同期1/5は検証器が物理残量`ledger_residual_all`でなく下方FINALIZE保留差を含む
  `unreconciled`を比較した誤検出。物理残量で比較するよう是正した。
- normal close後の未照合はt=448.800〜448.967の0.167秒、確定量34→35に対し
  着弾34→35が遅れた実事象。I7を緩めず、要約を
  `CLOSED_FORCED/late_finalize_after_normal_close`へ再分類し、未解決ゲートを維持する。
- 関連102テストと全pytest **6,161 passed / 13 skipped / 1 deselected / 0 failed**。
  v3固定snapshotのseg01は正式検証PASS。後着同期0/5、normal終了後未照合0/2、
  gross保存則0/53,620 side、global保存則0/26,811行、違法hard override 0。
- v2→v3は表示13/13列・通常timeline 44/44列bit-identical。条件5 sidecarだけ
  `closed_normal_unreconciled_count`/`last_close_reason`/`last_closed_status`の3/72列が
  意図どおり変化した。v1/v2は上書きしていない。
- 条件2は5/8完了、seg06/07実行中。v3はseg02〜04を1並列で継続。
  `src/production_config.py`未変更。
- 条件2の途中5/8では張り付き5.23%→4.67%、逆符号1.16%→0.96%、反転
  11→7回、急変19→15と改善。一方、seg05 game2（WIN★=1P）で最終表示が
  +60.745→+20.657へ弱まり、真の致死弱化1件。全8区間前だが必須条件0件を
  既に破ったため条件2単独は暫定NG。該当窓t=3736.200〜3744.667。
- 条件2は7/8完成。実行中の親shellファイルへCodexが上書き禁止チェックを追記したため、
  親が後半を読む時点で構文読取が途切れseg08だけ開始直後に終了した。完成済み7区間は
  無影響。旧`seg08.log`を保持し、同じ固定snapshot/flagsでcanonical seg08成果物を
  `seg08_retry1.log`へ再実行中。以後、実行中runnerは編集しない。
- 現在の3枠は条件5 v3 seg02、条件2 seg08 retry、条件4 seg01。

## 2026-08-26 17:32 JST — Gate 4条件3完了 / 条件5v2実測開始

更新者: Codex

- 最新判定: **`Gate 3R-6 PASS / Gate 4 MEASURING / production HOLD`**。
- WIN★勝敗根拠は全8区間で実試合109件を確定。UNKNOWN 3件は試合前または
  区間境界の0.033秒行だけ。seg04 game17は画像根拠つき2P勝利。
- 条件1は張り付き6.42%、逆符号1.95%、反転18試合/23回、急変27、
  決着逆方向23/109。条件3は張り付き9.93%、逆符号2.79%、反転17試合/25回、
  急変24、決着逆方向21/109、真の致死弱化0。急変・決着方向は改善したが、
  張り付きと逆符号が悪化したため単独採用は未決定。
- 条件5のlive会計はNone gap二重ID、max_sec/normal close後の遅延確定・決済、
  side wipe後の退役会計、古いclosed要約へのbackfill、hard override fail-open、
  検証器fail-openを回帰化して修理。独立再レビューはP1/P2残存なし。
- 最新全pytestは **6,160 passed / 13 skipped / 1 deselected / 0 failed**。
  条件5の正式snapshotは`_snapshot_cond5_codex_20260826_v2`。先行v1は上書きせず
  不採用監査証跡として保持。
- 現在は最大3並列で、条件2を1本、条件5v2を2本実測中。条件5を先に完了させ、
  実データ不具合があれば条件2・4の計測中に修理できる順序へ最適化した。
- user確認により条件1〜5をすべて正式比較する。これらは無条件に全機能を重ねる
  意味ではなく、条件5と条件3は旧累積器の二重会計を避けるため排他。条件2と条件5が
  ともに有効なら、条件5+ヒステリシスを全8区間で追加確認し、その統合構成で先頭5試合を
  動画化する。
- `src/production_config.py`未変更。本番ON、production登録、Gate 4採用判定、
  レビュー動画生成はまだ行っていない。

## 2026-08-26 15:29 JST — Gate 4条件1完了 / 条件3測定中 / 条件5実装済み

更新者: Codex

- 最新判定: **`Gate 3R-6 PASS / Gate 4 MEASURING / production HOLD`**。
- Gate 3R-6は先頭5試合の実データで、真の窒息2P t=223を`4/118`で検出、
  既知誤判定1P t=164.03〜164.73を`0/22`、待受画面2P t=18〜90.5を
  `0/1121`に抑えた。境界確定3件は敗北側と一致する。
- OFF独立3runは全3ペア`0/27`、ON/OFFは`0/27`、grossのみ/gross+deathは
  `0/42`で共通キーbit-identical。
- 正式境界への`game_idx`統合で増えた212表示行は、207/212が試合冒頭2〜3秒、
  ±100張り付き`0/212`、試合数不変のため受理した。
- 決着ホールドの絶対終了信号は、実連鎖したsideだけを対象にし、基準nextを
  `CHAIN_MIN_DISPLAY_SEC`経過後へ置き、CHAIN再進入・40点以上の得点増分を
  物理的な継続証拠として扱う。実窓の旧早期解除t=1713.033は再発0。
- 密なdisplay timelineは4,551行、最大間隔0.0333秒、欠測0。長いホールド3件は
  連鎖得点の継続または正式試合境界までの物理交換で、デッドロックではない。
- 全pytest: **6,107 passed / 13 skipped / 1 deselected / 0 failed**。
  dependency台帳の不足1件だけをテスト側で補い、production設定は変更していない。
- Gate 4はsettled更新行でなく、画面に実際に表示される全frameの密なdisplay timelineを
  正式母集団として、固定snapshot・manifest付きの条件1→3→2→4を測定中。
  集計器は張り付き・逆符号・反転試合・急変・決着方向・真の致死弱化を同一母集団で出す。
- 条件1は8/8区間完了。密な全表示7,033.3秒で、±100張り付き6.42%、生評価と
  逆符号1.95%、反転18試合（延べ23）、1秒150pt以上の急変27、gap異常0。
  旧14.0%等は母集団違いの履歴値として使わない。条件3（規模比較Aのみ）を3並列で測定中。
- 条件5はオンラインsnapshot、overlayアダプタ、ライブ/決着ホールド両経路の
  未解決hard-overrideゲート、独立sidecar、runner・検証器まで既定OFFで実装済み。
  関連239テスト、期限後決済修理後195テスト、勝敗根拠9テストは成功。
- 条件5実動画smokeはgross保存則0/3,600 side、重複生成0、回収不能なCLOSE後決済0、
  未解決中の違法±100適用0。max_sec直後の遅延決済2件は元の要約へbackfillした。
  短窓では正常CLOSE母数0のため、正式8区間で母数を確定する。固定snapshotは全pytest後に作る。
- 勝者根拠は死亡フラグを正解扱いせずWIN★パネル差分へ変更。自動差分が両側変化になった
  seg04 game17は実画面で28-29→28-30を確認し2P勝利と証跡つきで確定した。
- レビュー動画はuser決定により今は作らず、**Gate 4完了時の構成で先頭5試合を作る**。
  途中生成物は完成品として扱わず、既存ファイルも上書きしない。
- 2026-08-26 user決定により、レビュー動画の次は直ちに148動画学習へ進まず、
  **指標戦略壁打ち**を行う。旧モデル由来の指標選定を再監査し、死活・重複・組合せ・
  局面別有効性と最新のスポーツ統計学/ゲームAIの適用候補を整理してから学習条件を確定する。
- 当面の最優先はGate 4条件1〜5の完了とレビュー動画であり、指標の大規模実装へは分岐しない。
- `src/production_config.py`未変更、本番ON・production登録・採用判断は行っていない。

## 2026-08-25 12:19 JST — Gate 3-2b 再レビュー後

更新者: Codex

- 最新判定: **`Gate 3-2b REVIEW NG（局所修正合格・Gate全体未合格） / Gate 3-2c HOLD`**。
- 合格確認: P1-1、P1-2既知二重計上2経路、P2-1、P2-2、W38、W39。
- Codex関連再実行: **414 passed / 0 failed**。
- W39後の全pytest: **5,964 passed / 13 skipped / 1 deselected / 0 failed**。
- W39後の同一窓3run: NPZ全27キーとファイルSHA-256が完全一致。現行model52評価経路は決定化済み。
- Git状態: Claude修正は`19faa75`（P1/P2/W38）と`6c72386`（W39）として既にコミット済み。
  各報告の「コミットなし」は現時点では誤り。Codexはreset等を行っていない。
- 継続HOLDの理由:
  1. P1-3は設計だけで実装未着手。現設計は境界ワイプの「回数」しかなく、量を復元できない。
  2. `SETTLED_ECHO_MAX_SEC`が時刻だけで無関係なsettledも吸収する最小例をCodexが再現。
  3. zenchiの逆方向欠落6候補のうち、push観測0/190の2件は実欠落として未解決。
  4. Gate 3R-6は案Aだけでは届かず、画面連鎖中なのにstate=STABLEの遅れが本丸。
- 追加残課題: `src/old/indicators.py`の未seed経路が全pytest警告で可視化された。
  現行Gate 4経路とは別だが、legacy analyzer/overlayの非決定性として扱う。
- 交換台帳のオーバーレイ配線、`src/production_config.py`登録、PM100本番候補A/Bは引き続き禁止。
- 次の順序は `CODEX_TO_CLAUDE.md` 12:19節と `PLAN.md` Gate 3R-1〜6を正とする。

## 2026-08-25 14:20 JST — Claude制限後のCodex引継ぎ完了

更新者: Codex

- 今回Gate関連の実行中ジョブは0件。8/18開始のDL再取得補助と監視ループは
  別作業のため停止していない。
- P1-3: cap前gross累積カウンタと純差分アダプタを実装。境界ワイプは回数だけでなく
  消失量を累積するため、境界フレームを読み飛ばしても復元可能。
- K6: settled echoは時間だけで吸収せず、chain_count一致を必須化。掛け算式の
  成長観測済み連鎖では累積点一致も必須化した。
- P1-3実データ検収: zenchi本番30fps条件の9,000 frame / 18,000 sideで
  保存則残差0、最大残差0、未分類frame 0。境界ワイプ3,385個はledgerの
  `retired_unreconciled=3,385`と一致し、clamp loss 0。
- K7: 旧Q-01本番プローブそのものを包んで内部active/公開resultを同時採取。
  既存物理16連鎖は公開resultで真値 `(chain_count, total_score)` が16/16一致。
  問題の6候補もすべてinternal/public一致。既知mechanismの公開遷移157件で
  active/publicの値不一致0。activeのみ17件は全件`landing`/0点で会計対象外。
- K6: 時間だけの吸収を廃止した負例2本を追加。既存実データの吸収1/30は
  cc=3/累積1,260点が一致する真のechoで、修正後も維持。
- 局所回帰: 182 passed / 0 failed。全pytest: **5,976 passed / 13 skipped /
  1 deselected / 0 failed**（1,054.92秒）。
- `src/production_config.py`、オーバーレイ配線、既存監査成果物は未変更。

### 現在のGate判定

`Gate 3-2b / Gate 3R-4 PASS / Gate 3R-5 未着手`

Gate 3R-5の既定OFF配線へ進める。ただし`src/production_config.py`登録、本番ON、
PM100本番候補化はまだ行わない。次は既定OFF配線とtimeline列追加、その後に
Gate 3R-6のstate machine遅れを閉じる。

## 2026-08-25 08:10 JST — Codex独立レビュー後の最新状態

更新者: Codex

- Claude Codeは07:00にGate 3-2bの作業を終了。対象の実行中ジョブは0件。
- Claude基準: 全pytest 5,914 passed / 13 skipped / 0 failed。
- Codex独立確認: 関連624件pass、py_compile成功、ただし既存テスト未検出の
  P1を3件、P2を2件再現した。
- 交換エピソード会計は既定OFF・本番オーバーレイ未配線のため、現在の表示への
  直接影響はない。
- 掛け算式3フラグは`RECOGNITION_ADOPTED`で本番認識構成ONのため、
  FormulaStepAccumulatorのP2は修正・回帰確認が必要。

### 最新Gate判定

`Gate 3-2b REVIEW NG / Gate 3-2c HOLD`

オーバーレイ配線、交換台帳のproduction登録、Gate 4 A/Bへは進まない。
先に `CODEX_TO_CLAUDE.md` 2026-08-25 08:10節のP1/P2を修正し、
実データの再測定を通す。

### レビューNG

1. `simulate_fallback`のFINALIZEを拒否しても、同じ推定値がFIREとして台帳へ入る。
   52,150点のbaseline-only入力で`finalize_rejected_count=1`なのに
   `net_raw=745`を再現。
2. growthなしの`CHAIN_SETTLED`を即クローズした後に`SCORE_FINALIZE`が来ると、
   1物理連鎖から2つのchain_idが発行される。
3. `pending_uncapped`の純残量差分だけでは、同一フレームの新規受信と相殺の
   総量を分離不能。問題の「相手本線→自分本線」応酬で過小帰属しうる。
4. 大幅下方FINALIZEの再入力で`unreconciled`が二重加算される。
5. FormulaStepAccumulatorがinvalid/幕間でpendingを破棄せず、非連続2観測を
   「連続2フレーム」として確定する。

### 次の順序

`P1修正 → P2修正 → W38根治 → 115.7%再検算 → Gate 3-2b再レビュー →`
`既定OFF配線 → PM100/保存則A/B → モデルA/B/C/D比較`

以下の2026-08-24節は履歴として保持する。最新判断には上記を使う。

更新: 2026-08-24 18:18 JST / 更新者: Codex

## write lock

`RELEASED` — 14:18 JSTに実プロセス0、`ALL_DONE`、`BACKTEST_ALL_DONE`、
PM100 pytest完了を確認したため、旧write lockを解除した。

ただし既存差分のreset、checkout、stash、削除、上書きは禁止を継続する。
新規施策は既定OFFとし、`src/production_config.py` はユーザー承認前にONにしない。

## 実行中

- Claude CodeがQ-01〜Q-04のGate 0是正を継続中。
- Q-01の段同一性修正方針をfable reviewerで確認中。
- Q-02判定器、Q-03完全構成比較、Q-04仮想着弾案の新規成果物が生成中。
- CodexはClaude報告箱・成果物・実プロセスを監視し、完了後に独立レビューする。
- Codexの現行51指標監査が完了。61動画・全1,275ペアを評価し、未使用12動画で確認。
  詳細は `docs/CODEX_INDICATOR_AUDIT_2026-08-24.md` と `CODEX_TO_CLAUDE.md` 最新節。
- 局面別モデル追補を共有済み。現行は全局面共通モデル・共通ブレンドであり、
  Gate 0後に全局面/文脈付き単一/soft-gated 3モデル/ExchangeEpisode専門を比較する。

## 完了済み

- PM100全8区間のOFF/ON timeline dump生成。
- PM100修正の単体テスト: 31 passed。
- 指摘場面 seg01 game2: OFFは約1.6秒で +100→-99、ONは +100→+25を維持。
- 掛け算式10ケース、偽イベント4走行、8動画×3条件バックテスト完了。
- OFF baseline / OFF final / 先行OFFのMD5一致。
- ClaudeがCodex品質監査Q-01〜Q-04を受諾。Q-01は独立テストで再現。
- fable architect/reviewerは交換仕様を条件付き不合格とし、Gate 2前の仕様修正を要求。
- 全pytest: 5676 passed / 13 skipped / 2 failed。速度失敗は低負荷単独PASS、
  おじゃまダメージ失敗は既存意味論不具合として再現。
- 指標監査: 現行モデルは序盤AUC0.5256 / 中盤0.5506 / 終盤0.7843。
  forecast系3列はほぼ完全重複かつ95.1%が0。時間差交換にはExchangeEpisode会計が必要。

## PM100全8区間結果

- 急変: 79回 → 40回。
- 反転試合: 47 → 34。
- 決着方向誤り: 10 → 7。
- ±100張り付き: 14.0% → 18.3%へ悪化。
- 生モデルと逆符号の張り付き: 3.3% → 5.2%へ悪化。

結論: 現行A+Bは指摘場面には効くが全域採用不可。既定OFFを維持する。

## Gate判定

`Gate 0 継続` — Q-01〜Q-04、w2の旧`formula`混入、v51 ON 99.391%、
母集団差、交換仕様レビュー指摘を閉じるまでGate 2統合・本番採用へ進まない。
