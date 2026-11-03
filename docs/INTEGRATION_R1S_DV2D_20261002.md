# A: exev への R1s / D-v2d 統合

対象は `D:/puyo_analyzer/wt_integ`、`claude/integrate-r1s-dv2d-20261002`。
基点 `97a9fcd`。最終判断・本番登録の承認権限は統括 Claude に留保する。
ユーザー指定の登録を実装したもので、新しい採用判断を行ったものではない。
作業開始 2026-10-02 19:19:54 JST、最終検証完了 19:57:36 JST (37分42秒、コミット前)。

## 統合と衝突解消

- `af583a5` を merge: `9c0ac0e1`。衝突なし。
- `42d82b1` を merge: `9602903`。
- `replay_exchange_event_20260926.py`: 静止 G_fe 引数と D-v2 系列引数を併存。
- `visualize_advantage_overlay.py`: 静止 G_fe・切替平滑化・D1〜D4/H1 引数を併存。
- `exchange_event_cli.py`: 候補上限検証を維持し、本番フラグ展開後に E35 依存・排他検証。
- `exchange_event_overlay.py`: 両系列の引数と候補上限検証を保持。検証の実行順を変えず、
  初期化補助関数へ移してコンストラクタを50行以内に収めた。
- 仕様判断を必要とする競合はなかった。

`models/exchange_event_v1_r1_hgb/` は Git 管理済みのためコピー不要。
G_fe.joblib の SHA-256 は manifest と一致:
`dda548efb86aa0b6c1e71c079239cb483fc98d55ae418d80fa9de3b6de6c3da4`。

## 必須修正

通常運転では `PostCounterDeathBound.audit`、`MatchHistoryIntegrity.audit/games` を
試合境界で破棄する。保存再生だけ `retain_audit=True` とし、全試合の監査原票を維持する。
`ExchangeRecord.history_integrity` に試合の証拠を保持し、辞書破棄後の保存・ライブ側の
dataclass 封印でも失わない。H1 OFF の既存 JSONL には新しい null 項目を追加しない。

H1 が死亡を取り消す際は `single_death_proof` と `post_counter_bound` の公開 `dead` も
False にする。元の計算結果は `computed_dead` と内部 proofs に保持する。
辞書を複製して処理し、キャッシュ・既存 record を遡って変更しない。
コンストラクタのエラー文には E35 とその前提3フラグを明記した。
機能の API/CLI 既定は OFF のまま維持する。

## production_config の登録差分

採用日は全件 `2026-10-02`。根拠は依頼文を一字も変更せず登録した。
D-v2d の共通根拠だけ `DV2D_ADOPTION_REASON` にまとめ、各項目は個別の AdoptedFlag とした。

| バケット | 追加フラグ |
| --- | --- |
| EXCHANGE_EVENT_ADOPTED | `--death-input-integrity` |
| EXCHANGE_EVENT_ADOPTED | `--score-attribution-v2` |
| EXCHANGE_EVENT_ADOPTED | `--post-counter-current-puyo-integrity-v2c` (E35″) |
| EXCHANGE_EVENT_ADOPTED | `--death-history-integrity` |
| EXCHANGE_EVENT_ADOPTED | `--exchange-event-static-gfe-dir models/exchange_event_v1_r1_hgb` |
| PLACEMENT_RECONCILE_ADOPTED | `--falling-puyo-integrity` |

評価側4フラグは `enabled_flags('v2d')` と同じ。
D4 は描画CLIから `generate` → `RecognitionPipeline.load_default(enable_falling_puyo_integrity=True)`
へ届くバケットを選んだ。rt の `run_live_pipeline_20260928.build_command` も
`--production-exchange-event` を渡して同じ描画 main を呼ぶ。
RECOGNITION_ADOPTED は CLI 名をそのまま kwargs に変換するため、この enable_* 名と一致せず、
既存の明示配線とも重なる。こちらには入れない。

## 検証結果

| 検証 | 結果 |
| --- | --- |
| 関連回帰 | 1,463 passed / 1 skipped / 5 deselected / 0 failed、203.91秒 |
| 実描画 main → argparse → generate → overlay | 本番登録そのものを使用。D-v2d ON、静止 HistGBC / 仮想着弾 LR |
| 実描画 main → load_default | D4=True を確認。映像/CNN I/O のみスタブ |
| E35″ 正の対照 | 未確定な手持ち前置では死亡未証明、観測後省略では死亡成立 |
| 実終了センサ結合 | CHAIN の基準確定 → NEXT移動まで更新。省略解除・キャッシュrevision更新を確認 |
| 保持量と封印 | 100試合境界、通常1試合 / 保存再生100試合。実tracker保存後も前試合の開始証拠を保持 |
| H1整合 | 公開dead取消、計算証拠保持、既存辞書の不変を確認 |
| rt計測接続 | 原本の一時コピーにB用パッチを適用し3項目の保持量取得に成功。原本SHA不変 |
| モデル資産 | SHA-256 が manifest と一致 |

初回は1,461件成功・2失敗。統合によるコンストラクタ52行と、新規テストのI/Oスタブ不足を修正。
最終ログは `logs/integration_r1s_dv2d_20261002/pytest.log`。
skip は E2_SHORT_ARTIFACTS 未指定の既存検査。除外5件は既定OFFの全長5記録再生であり、
その互換テスト自体は新採用フラグを外した旧構成を検査するよう更新した。
前系列の全長5記録は269件成功の検証内で全通過している
(`logs/eval_set/falsedeath_fix/v2d/REPORT.md`)。今回は二記録の短区間を再検証した。

全リポジトリの pytest は今回再実行していない。
参照できる全体成功記録は2026-09-04の7,729件PASS (DECISIONS.md)。その後の
2026-09-20修復では、同居問題のある22ファイルを隔離して合算する契約が決まり、
全体再走はせず197件の局所CPU検証を実施している (PYTEST_REPAIR_2026-09-20.md)。
今回の1,463件は全体件数と分母が違い、全体の非退行を証明したとは扱わない。

## 保存記録の比較

4構成を分けて実行した: 旧認識+R1s、D4記録+R1s、D4記録+v2d、D4記録+R1s+v2d。
各構成は実描画CLIで構築した overlay に保存観測を流す。元動画の再認識・描画は行わない。
静止 G_fe の実呼出しは HistGBC、review の仮想着弾1,440呼出しは LR である。

| 記録 / 表示区間 | R1s保存表示を対照で再現 | v2d保存表示を対照で再現 | 統合版とR1sの表示値差 | 統合版とv2dの表示値差 |
| --- | --- | --- | --- | --- |
| review / 2579.05〜2769.05秒、5,701行 | 全表示列一致 | 全表示列一致 | 3,701行、最大70.4080点 | 3,799行、最大16.1950点 |
| q / 0〜120秒、3,601行 | 全表示列一致 | 全表示列一致 | 0行 | 3,598行、最大88.6484点 |

review の逐次差分: D4入力差だけで表示値3,205行・由来99行、そこへD-v2dを足すと
表示値1,383行・由来87行が変化する。v2dへR1sを足した差では由来の変化は0行。
行集合は重なるため、これらの件数・最大値は加算しない。
qの検証区間は通常静止で、D4/D-v2dを足した差は0。R1sモデルによる表示値差だけである。
q第14試合は今回の短区間には含まれない。
短区間ではH1による取消の実発動は0件で、取消整合性は専用結合テストが担保する。

最終原票: `logs/integration_r1s_dv2d_20261002/native_m0/RESULT.json`。
最初は検証器でM0を1スレッドにし、保存表示と最大1.39e-5点の微小差が出た。
本来のM0設定 (2スレッド) に戻すと両記録・両系列の全表示列が完全一致した。
この診断で本番コードのスレッド設定は変更していない。BLASは全実行で1。
再生 nice=10、回帰 nice=19。各起動前と各再生後に空きメモリ6GiB以上を確認した。

## A/B の境界

A の基点には `src/phase_j/live_memory.py` が存在しないため、rt自体は取り込んでいない。
A側の `src/exchange_integrity_memory.py` と、B適用用の
`docs/patches/rt_live_memory_dv2d_20261002.patch` を納品する。
パッチは適用可能性と実計測を検証済みだが、rt原本には未適用。
実ライブでの接続はBの統合時に適用する必要がある。既存worktreeのファイルは変更していない。

再実行:

```bash
bash scripts/run_integration_suite_20261002.sh
bash scripts/run_integration_checks_20261002.sh -m scripts.verify_integration_r1s_dv2d_20261002 --native-m0 --output logs/integration_r1s_dv2d_rerun
bash scripts/run_integration_checks_20261002.sh -m scripts.verify_integration_r1s_dv2d_20261002 --audit-only
bash scripts/run_integration_checks_20261002.sh -m scripts.verify_integration_memory_patch_20261002
```
