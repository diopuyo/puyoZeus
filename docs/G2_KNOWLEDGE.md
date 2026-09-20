# G2で残す設計知識と検証の教訓

対象はPR #26のvideo38診断経路。これは本番採用仕様書でも、全動画での精度保証でもない。2026-09-14時点でA40の動画末尾到達と修復箇所通過はユーザー受入済みだが、終了時I/O障害による全保存監査は未完である。事実・限定的な設計・未検証を分けて読む。

実績の要約は[G2修復レビュー](G2_REPAIR_REVIEW_2026-09-14.md)、ファイル単位の収録理由は同PRの必要性台帳を参照する。ここでは一回限りのrun番号ではなく、次の開発にも再利用できる知識を残す。

## 1. 現在盤面・予測盤面・勝率を混ぜない

観測して確定した盤面、観測条件付きで推定した連鎖後盤面、その盤面から学習器が出す勝率は別の値である。予測候補が一意でも、その値を観測真値や較正済み勝率へ昇格させない。

- 既存の確定盤面評価は両者STABLEのconfirmed_boardを基準とする。
- G2の確率的な将来評価入力は別の診断経路。旧M1の資格を満たしたことの代わりにはならない。
- 隠し段は既存の確率分布を保持する。可視GTと隠し段推定を同じ正解分母に入れない。
- 可視盤面が予測支持集合に含まれることと、予測確率の較正が良いことは別々に測る。

隠し段を検査時だけ0化すると、実際に保存した盤面と別のものを照合してしまう。保存側と検査側で、同じ時点・表現・盤面の出自を使う。

参照：`data/verify/g2_second_terminal_arrival_2026-09-14_v3/terminal_saved.py`、同ディレクトリの`test_saved_raw_tail.py`。

## 2. 到来・消費通知・物理反映は三つの出来事

NEXTへのツモ到来、Nativeの消費通知（ACK）、盤面への物理反映は同時とは限らない。ACKだけを「着手済み」の唯一の指標にすると、通知が遅い終端で有効な候補を失う。一方、遅いACKを新しい着手として扱うと二重反映する。

各tokenについて到来・ACK・反映を別々に持ち、物理反映は一回に制限する。原FIFOは観測源として扱い、検査を通すためにACKを追加したり書き換えたりしない。

A40の2Pでは到来11・反映11・ACK10で、未ACK1を保持した。この差は自動的に会計不正を意味しない。終了後8更新では追加ACK・反映とも0だった。1Pは7・7・7である。

参照：`data/verify/g2_arrival_ack_candidate_2026-09-12_v1/ledger.py`、`data/verify/g2_second_terminal_arrival_2026-09-14_v1/second_arrival_state.py`、同ディレクトリの`test_portable_arrival_contract.py`。

## 3. 候補ゼロを安易な補完で解消しない

通常の配置列挙で支持がゼロになった場合、最初に原消費列、複数消費、ちぎり、観測資格の接続を調べる。CNNの推測を確定盤面へ上書きして失敗を隠さない。

G2で追加した終端回復は、同callの観測資格、連続した予告情報、自己発火・相殺がないこと、最大2手、初回30個着弾、全候補の終端性、観測一致盤面の一意性という限定条件を持つ。特定frameだけの例外ではないが、非終端や2手超へ一般化した仕組みでもない。

候補質量1は「この仮説と観測条件の下で一意」を意味する。認識信頼度1、勝率1、将来着弾保証を意味しない。

参照：`data/verify/g2_second_terminal_arrival_2026-09-14_v3/observed_terminal_drop.py`、同ディレクトリの`session_selection.py`。

## 4. 得点増加を連鎖・相殺と同一視しない

ソフトドロップ等の得点増加と連鎖消去による得点を分ける。A36では予告資格の検査が全フレームの得点差0を要求し、非連鎖の小得点を誤拒否した。人工fixtureの得点が全0だったため、この違いを検出できなかった。

修復では既存の得点分類と連鎖・origin・event信号を再利用した。「+1なら無条件に無視」「毎frame40未満なら許可」のような近道にはしない。累積値・未知値・混在得点・予告中断も扱う。大きな落下得点の一般化は未検証である。

ゲーム知識で結論が変わる部分は、OCR誤読などと断定する前に既存ロジックとユーザー知見を照合する。レビューの指摘も仮説であり、正解ではない。

参照：`data/verify/g2_second_terminal_arrival_2026-09-14_v3/warning_source.py`、同ディレクトリの`test_warning_score.py`、既存`src/scoring.py`。

## 5. Session開始前の履歴と、両側の時点を揃える

評価Sessionの生成前に始まった原originや連鎖の履歴を捨てると、後から現在盤面だけを渡しても正しい予測初期状態を作れない。履歴の蓄積・所有者・移管・sealの順序を明示する。

自分側の連鎖後予測と相手側の現況を比較するときは同じcutoffに揃える。古い対向盤面を無言で現在値として使わず、READY/HOLDとその理由を保存する。

参照：`data/verify/g2_second_prefix_runtime_2026-09-14_v40/early_runtime.py`、同ディレクトリの`early_probability_capture.py`、`data/verify/g2_history_prediction_diagnosis_2026-09-13_v1/session_runtime_binding_candidate.py`。

## 6. 型の合成順・loader・保存consumerも修復対象

正しいクラスを書いても、実creatorの前後で包装順を誤ると最終Session型や元の監査契約が合わない。試験用に直接生成した型のPASSだけでは実creatorの動作を保証しない。

確認する一単位は、入口、before_create、実create、最終型、origin binding、更新callback、保存consumer、close、alias/参照復元まで。producerだけ新型で保存consumerが旧数学のまま、という非対称な更新を避ける。

動的aliasを解放する時点にも注意する。終了後検査がまだ必要なクラス・module・bindingを参照するなら、先に解放してはいけない。ただし別scopeの参照を借りて失敗を回避するのも不可。

参照：`data/verify/g2_second_prefix_runtime_2026-09-14_v40/owned_adapter.py`、`data/verify/g2_a38_postrun_2026-09-14_v1/creator_terminal_checks.py`、`data/verify/g2_a39_postrun_2026-09-14_v1/probe_terminal_creator.py`。

## 7. inactive、同世代終了、別世代resetを区別する

通常のinactiveをRegistry退役や別epochへの移行と同一視しない。終了らしい表示だけで更新をskipすると、後続completedや保存処理の契約を壊す。

G2は原enqueue/step・同scope・世代・終了票・元例外の有無を照合して同世代終了を認証し、盤面と台帳を凍結する。終了後の時計が進んでも、凍結盤面を新しい確定観測として発行しない。

A39の停止は、終了後に台帳clockを凍結した一方で、後続予測readerが現clockとの一致を要求した接続不整合だった。A40は所有・時計の通常検査を残したまま、認証済み終了だけを明示HOLDにした。

この確認は別epoch退役後の再入、複数Sessionの一般化、live hookや将来供給の認可を意味しない。

参照：`data/verify/g2_a38_postrun_2026-09-14_v1/terminal_selection.py`、`data/verify/g2_a39_postrun_2026-09-14_v1/terminal_prefix_reader.py`、同ディレクトリの`test_terminal_reader_limits.py`。

## 8. 未評価は未評価として保存する

旧raw-STABLE M1の必要入力が不足したときはunsupportedを保存する。新しい確率経路が動いたことと、旧M1の数値採録に成功したことを混同しない。A40の旧M1は採録0である。

HOLD、unsupported、試合外、欠測を全部0点やEVENへ潰すと、後段学習で意味が変わる。次の比較ではそれぞれの分母と扱いを事前に固定する。

参照：`data/verify/g2_legacy_m1_compatibility_2026-09-14_v1/compatibility.py`、`data/verify/g2_second_prefix_runtime_2026-09-14_v40/review_m1.py`。

## 9. 動画末尾到達・正常終了・保存監査PASSは別

A40はframe36900まで到達したが、終了時のconsumer_publication書込でI/Oエラーとなり、原entryは1だった。guard/supervisorの終了0で子の異常を覆い隠さない。

- 構文が通るJSONにも、途中までしか保存されていない可能性がある。
- COMPARISONのequal要約がtrueでも、元ROWS破損やstatus欠測があれば全監査PASSではない。
- 保存後の数学再検算は有効な追加根拠だが、原runをexit0へ書き換える理由にはならない。
- 容量不足は有力だったが、Errno5だけで原因を一意に断定しない。

RAM/VRAM保護とは別にディスク余力と終了時の追加書込量を考える必要がある。新規検証成果物のD保存は承認済みだが、凍結runnerの出力先を変更済みという意味ではない。容量監視の実装完了も主張しない。

参照：`data/verify/g2_a40_postrun_2026-09-14_v1/audit_saved_files.py`、同ディレクトリの`review_second_saved.py`。原結果とD上の追加検収票の区別は修復レビューを参照。

## 10. 保存票があることと、厳密に再開できることは別

盤面や原Jを保存しても、SM内部履歴、OCR/HSV適応状態、未完call、FIFO、Registry所有関係、保存cursorが揃っていなければ、生存pipelineと同値なcheckpointではない。

同一process内の参照保持を、process終了後の復元能力と呼ばない。厳密再開できない場合、前半の状態再構築が必要になることを明示する。別runの同frameを無条件に新runの真値へ流用しない。

既存PASSは版・入力・適用範囲を紐付けて保持し、後続の失敗だけで全体を再試験しない。必要な再構築と、合格項目の再認定は分ける。

## 11. レビュー・CPU・実動画の役割を分ける

推奨順序は、実分岐の設計反証、最小CPU反例、最小修復、旧反例と正常対照、実creator/保存/解放、実動画でしか得られない情報の取得である。

人工空FIFO、全0得点、人工初期履歴、no-op basis、MENUだけの成功は、原動画の陽性事例の代わりにはならない。試験には人工部分・未到達部分を必ず書く。

独立レビューも万能ではない。今回のPR選別では、旧v33/v38削除案に対して実importが見つかったため採用しなかった。grepの対象がgitignoreで除外されていた場合、「参照なし」という結論は成立しない。

CPU/GPU使用率そのものを成果指標にしない。主工程を遅らせない独立検査を並行し、重複再走・無変更再レビュー・空回しを避ける。

## 12. 試作の依存をPRへ丸ごと累積しない

実load一覧、固定SHA対象、テスト、歴史資料は別の集合である。旧runで読まれたことだけを現在の必須性にしない。逆に、古い日付だけを理由に動的import先を削らない。

各収録ファイルについて、どの入口または回帰から必要になるか、実loadか明示参照か、外部資料を必要とするかを残す。directory名の登場だけではfile単位の必須証明にならない。

mainから専用branchを作り、既存dirtyと検証成果物を保持して明示選別する。PRからの除外とローカル資料の削除は別操作である。試作runtimeの収録を本番採用と呼ばない。

## 次工程に残る問い

- 複数動画での支持率、HOLD/unsupported率、GT分母、遅延許容値。
- 別epoch退役・複数Session再入と、終了後のlive hook寿命。
- 条件付き予測確率の較正、統合勝率の精度、M2の残監査。
- D保存runnerの実接続、容量監視、欠測した全保存監査。

これらをG2の局所修復や最小CPU PASSで閉鎖しない。G3実走・学習・本番採用・自動mergeは、今回の資材整理と知識保存には含めない。
