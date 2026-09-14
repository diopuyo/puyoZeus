# G2修復・実動画完走のレビュー

## 結論と受入範囲

最新のマージ承認：ユーザーはG2限定修復のマージまでを承認した。診断・検証用ソースと知識の統合であり、全保存監査PASSや本番採用ではない。受入条件・実テスト結果・残制約は[G2限定マージの受入](G2_MERGE_ACCEPTANCE.md)を参照する。以下のDraft提出時点の制約は品質上の事実として保持し、古い「マージしない」という停止境界だけを今回の明示承認で更新する。

ユーザーはA40の動画完走と修復箇所通過を完了扱いとし、D保存への変更とPR提出を承認した。対象はvideo38のG2修復であり、統合勝率の精度・較正・複数動画への一般化・本番採用ではない。

A40はframe 29052〜36900をstride 2で処理し、最後の両側stepまで到達した。終了8更新も保存済み。全保存処理の終了コードは1であり、無条件の品質ゲート合格や正常exit0とは呼ばない。終了時I/O障害による未回収の監査を明示して、レビュー用Draft PRとして提出する。保存障害だけを理由に全動画は再走しない。

## 修復内容

- 発火前の原イベント履歴と確率入力を評価Sessionへ移管し、連鎖後予測と対向側の現況を同じcutoffへ結合する。予測盤面を現在の確定盤面へ昇格させない。
- 通常手・複数消費・ちぎり後の候補を既存の配置列挙と確率盤面へ接続する。到来、Native ACK、物理反映を分離し、未ACKを保持して一回だけ反映する。
- 観測資格が成立する限定的な終端着弾を既存数学で回復する。原FIFOや得点を改変せず、+1落下得点の誤拒否と隠し段を0化した保存照合を是正する。
- 旧raw-STABLE M1の入力不足をunsupportedとして保存する。新しい確率機構の動作と、旧M1の数値採録成功を混同しない。
- 通常inactiveを別世代resetと同一視せず、同scopeの原enqueue/step/終了文脈で終了を認証する。盤面と台帳を凍結し、架空のACKや物理反映を追加しない。
- 終了後の予測readerは原所有/世代/時計チェックを維持し、認証済みの終了だけ明示HOLDにする。最後のSession型・loader接続と元保存consumerの参照を保持する。

## 実動画と保存後の検収

- A40実走：3107.94秒、frame36900到達。原entry1、resource guard0。旧A39のprojection_live_finished_clockは通過。
- 1P：終了36886〜36900の8更新を原Jと照合。7到来/7反映/7ACK、追加ACK0・追加反映0、終了票のclosed/末尾/receipt一致。後処理48.27秒。初回後処理は検査器のmodule adapter選択ミスでAttributeErrorとなり、修正したv2で確認した。これは元動画runの障害とは別で、初回失敗票も保持した。
- 2P：元の全prefix数学・着弾数学と原Jを使って36900まで保存再計算。11到来/11反映/10ACK、未ACK1を保持。終了8更新の追加ACK/反映0、548時点のpending履歴。後処理83.00秒。
- 終了予測：HOLD8件、理由match_ended_scope_frozen。予測captureのlast_frame36900、stream/capture closed、元例外なし、cleanup_errors空。close処理の確認であり、live hook検証と将来供給の認可は未実施。
- 旧M1：schedule.last36900、pendingなし、採録0。数値未評価のままであり、M1品質成功へ読み替えない。
- 既存A37/A38の会計・予測入力・可視GT支持・metadata差0はその版/範囲の根拠として保持する。A40の欠測監査へ無条件に転記しない。

## CPUと独立レビュー

- 元readerの最小時計反例1件。修復候補14件、追加境界/所有5件を確認。
- 原OriginCapture→終了HOLD→原reference/root保存→親→closeのCPU1件。初期履歴/pair/親は人工である。
- A39の実終了J/context/flags/台帳→原資格/Scheduled.completedのCPU1件でWAIT・保存0・pendingなし。basisはno-opである。
- A40の実creator/最終型/reader実bind/旧4bind/保存解除をCPU確認（117.11秒）。人工初期履歴を実動画完走とは呼ばない。
- Fable333設計、334差分、335新証拠検収を実Read確認。限定CPU接続の既知反例なし。これ自体は全G2/本番GOではない。
- Fable336は保存後検収票とPR説明を独立確認し、提出を止める矛盾なし。比較要約の存在、検査器の初回失敗、close確認の限定範囲の3点を説明へ反映した。
- PRに含む最小ポータブル契約試験は、動画/モデル不要の `data/verify/g2_second_terminal_arrival_2026-09-14_v1/test_portable_arrival_contract.py`。その他の統合fixtureは私有動画・原票・凍結snapshot・モデルに依存し、クリーン環境の全件CI成功を保証しない。

## 保存障害と未回収の監査

- 原エラー：終了処理 `g2_publication_consumer_runtime_2026-09-09_v1/runtime_adapter.py` のconsumer_publication.jsonl書込で `OSError: [Errno 5] Input/output error`。
- 発見時C空き約89MB。容量不足が最有力だが、Errno5だけで因果を断定しない。RAM監視にディスク容量監視がなかった。
- JSON構文監査で `POSTCOMMIT_CONSUMER_ROWS.json` の途中切れを確認。元metadata比較器は `stream_incomplete_or_invalid_object` で拒否した。欠測を補った合格票は作っていない。
- `POSTCOMMIT_CONSUMER_COMPARISON.json` は構文上有効でequal要約は全true、death_equalもtrue。ただしROWS破損と同一終了処理の産物であり、statusも欠けるため監査合格として採用しない。
- 未生成：PRIVATE_CONSUMER_STATUS.json、PRIVATE_CONSUMER_COMPARISON.json、POSTCOMMIT_CONSUMER_STATUS.json、通常のTERMINAL_SAVED_REVIEW.json。2P保存数学は別の後処理票で確認したが、通常の成功票を捏造してはいない。
- 原Cファイルとentry1は不変更。新しい検収票は `D:/puyo_analyzer/verify/g2_a40_recovery_2026-09-14_v1/` に保存。run_idは原Cの値を保持する。
- 初回提出は保存完全性の全監査が未完のためDraftとした。現在の限定マージ判断は上記受入文書で行い、本番採用や全保存監査の成功とは区別する。回収不能の比較票と新しい保存・容量監視の実装は未完のまま明示する。

## 提出範囲と停止境界

### 収録範囲の追加監査（PR提出後）

最新：707件をfile単位で確認し、706保持・旧v2選択器1件除外へ判定した。内訳と根拠は[必要性確認](G2_SOURCE_NECESSITY.md)と[全件台帳](G2_SOURCE_NECESSITY.json)、再利用する設計知識は[G2ノウハウ](G2_KNOWLEDGE.md)へ永続化した。未分類保持はなくしたが、本番依存だけの最小構成・全動的分岐網羅・全保存監査PASSを意味しない。以下の「707全件未証明」は前回整理時点の履歴。

A40の固定source、回帰試験、採用した保存後検収を起点に参照を監査した。不採用の単発診断 `g2_a40_postrun_2026-09-14_v1/probe_metadata_unit.py` と `probe_metadata_unit_v2.py`、および旧 `g2_second_prefix_runtime_2026-09-14_v35`・`v36`・`v37` の各10ファイル、合計32件をPRから除外した。ローカルの原ソース・失敗票は保持し、監査失敗の事実は本書から削除しない。

旧版のruntime/freezerは日付だけでは削除できない。A40の `freeze_target.evidence()` はA39の固定票が指す旧ソースのSHAを確認するが、A39 freezer自体を再帰実行しない。除外したv35〜v37はA40固定集合に含まれず、逆参照は旧freezerの歴史証拠と旧launcher内だけだった。旧v38以前の固定票の再発行はこのPRの対象外で、必要なら保持したローカル資料を使う。現行A40/A39固定集合は全保持する。

独立Opusレビューのv33/v38除外案は親確認で不採用とした。`g2_second_pending_prefix_2026-09-14_v3/postrun.py` がv33のtarget_entryをimportし、`g2_legacy_m1_compatibility_2026-09-14_v1/test_saved_compatibility.py` はv38のreview_m1を読むためである。残707ファイルには動的directory参照を安全側に保持した部分があり、全ファイルが最小構成で必須という証明ではない。最小化の未証明部分を残したまま、全件必須・マージ準備完了とは主張しない。

最終A40・修復依存・必要な検査sourceを選別する。既存tracked dirty、動画、モデル、巨大JSON/JSONL、外部snapshot、rawレビュー、過去失敗成果物をPRに混入しない。私有環境の依存closureを完全にポータブル化した変更ではない。

PR本文に加えて修正一覧コメントを投稿し、G3は既存6動画/比較版/GT分母/遅延許容/開始条件の文書引継ぎで停止する。自動merge・本番採用・学習・G3実走は行わない。再開時は6動画を一巡してから結果と優先順位をユーザーと相談する。

新規検証データはD保存へ変更したが、凍結A40の出力先や原run_idを遡って変更していない。次の実走runnerをD対応したという主張でもない。
