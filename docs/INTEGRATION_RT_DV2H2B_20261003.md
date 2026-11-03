# B: D-v2h2b リアルタイム統合

対象: `wt_integ_rt` / `claude/realtime-integrate-r1s-dv2d-20261002`、基点 `922c69d`。
Aの `b3c5d1b` をマージ。作業開始はA/B共通で2026-10-03 03:21:43 JST。
Aの資源検証追補 `4d9c654` も追加マージした。
最終採否は統括 Claude に留保。push・ゲーム画像の追加コミットは行わない。

## 変更と競合解消

11ファイルの競合を解消した。
追補マージでは検証ランナー1ファイルだけが競合し、出力先をA/Bで決めた後に
mkdirするrt側の順序を維持した。Aの資源プラグイン・テスト修正は同内容で一致。

- `.gitignore`: rtの機器別較正設定の除外と、ゲーム画像の除外を両方保持。
- `exchange_event_tracker.py`: rtの単調な通しIDと `record_mapping` による保存・封印の
  共通化を維持。試合辞書をclearした後もrecordのH1証拠を保存する。
- `exchange_event_cli.py`、`exchange_event_landing.py`、`exchange_event_overlay.py`、
  `exchange_history_integrity.py`、`exchange_post_counter_bound.py`、`production_config.py`:
  Aの新構成を採用し、rtの監査保持・境界clearと両立。
- `test_exchange_event_production.py`、`test_falsedeath_v2.py`、
  `test_integration_r1s_dv2d.py`: Aの本番構成とkeyword-onlyに追従した期待値を採用。

rtの共有overlayにあったtracker更新前の `refresh_current_observation(self)` は削除。
`SplitExchangeOverlay` の更新前呼出も削除し、更新後の `_observe_landing` から
`refresh_current_observation(self, result, t_sec)` を実行する。
通常の通知workerは `NotificationExchangeOverlay` → 共有overlay → landingを通り、
保存再生と同じ順序・同じresultでE35へ入力する。

MatchHistoryIntegrityの全引数、PostCounterDeathBoundのearly_exit以外はkeyword-only。
src/scriptsの全呼出を確認し、渡す引数は全てkeyword指定。
テストの `MatchHistoryIntegrity(True)` は位置引数の拒否を検証する意図的な負対照だけ残す。
H1′は前試合の非ゼロ得点を取得してからgames/auditをclearする。
通常liveの監査2,048件上限、保存再生の `retain_audit=True` は維持する。

production_configはAと完全一致。追加4フラグ・採用日2026-10-03・指定根拠は
`docs/INTEGRATION_DV2H2B_20261003.md` に記載した。

## 検証

- 初期のworker配線・H1′・E35・統合: 77 passed、31.23秒。
- 保存再生・ライブ時系列の最終4ケース: 4 passed、13.55秒。
  実gzip記録へ保存→read_recordsの復元入力と、ライブ通知入力で照合履歴が一致。
  式・表示得点の3段、2増分が一致。更新前照合・result欠落・時刻欠落も検出する。
  両live経路で試合境界clear後のH1′受理が保存再生と一致。
- live一式と関連回帰: **2,250 passed / 1 skipped / 5 deselected / 0 failed、1,028.28秒**。
  skipはE2_SHORT_ARTIFACTS未指定、除外5件は既定OFFの全長保存記録再生。
- 変更対象の最終確認: **103 passed / 0 failed、165.67秒**。
  `scripts/run_rt_dv2h2b_final.sh` で実行し、開始・終了ともnice10・BLAS各1・
  空きメモリ6GiB以上を確認した。件数は上の回帰と重複するため合算しない。
- 構文・対象関数50行以内・src/scripts全体のコンストラクタ位置引数ゼロ:
  `logs/rt_dv2h2b_20261003/STATIC.json` ですべて通過。

## 実動画

前回と同じ `video_zenchi_c0BQoMJwwQU.mp4` の2579〜2650秒、warmup1秒。
`--source video --realtime --cpu-threads 1 --evaluation-nice 10 --cnn-device cpu`。
nice10、BLAS各1、各起動前の空きメモリ6GiB以上を検査。
既存動画は読み取り利用し、画像は出力しない。

広域pytestはnice10で起動したが、既存の学習初期化テストが途中でpytest自身を
nice19へ変更した。検出後、このpytestプロセスだけをnice10へ戻した。
実動画workerはこの別プロセスの副作用を受けず、全標本nice10を確認している。
Aの追補でテストの優先度変更を抑止し、資源監査付き最終検証へ引き継ぐ。

初回原票は `logs/rt_dv2h2b_20261003/initial_live/` に保持した。
1,066通知、全8フラグON、式・表示入力あり、エラー・再起動0。
ただしこの区間の死亡証明監査にはcompletion_prefix_checkが現れなかった。
「未観測以外」の確認に不足するため、実際に呼ばれるprefix_checkの戻り値を
変化時に記録する計装を追加し、同一区間を `live/` へ再走した。
評価入力・判定・公開頻度は変更せず、計装だけを変更した。

通知時間は実workerの `NotificationExchangeOverlay.update()` の入口から戻りまで。
H1・段観測・照合結果の計装コストを含み、通知ログ・RSS標本の書込みは含まない。
RSSは同じ評価workerで30通知ごとに標本化する。
回帰試験と同時実行した参考値であり、専有環境の性能比較や長時間リークの否定ではない。

最終実走: 終了コード0、評価エラー・処理エラー・worker再起動はすべて0。
全8フラグON、全ての観測変更にresultが渡り、式・得点表示の段履歴を実受理した。
照合結果の変化は168件: `recent_mismatch` 162件、`recent_unobserved` 6件。
これらは独立した連鎖の数ではなく、実prefix_check戻り値の変化を記録した件数。
例: 6→7段の観測増分は式・表示とも6,500点。発火時の元予測に該当段が無いため
照合不一致として抑止する。実動画で照合の陽性受理まで確認したという意味ではない。

H1′は2637.3667秒の境界で両側0読みを期限内に観測（timely=True）したが、
STABLE盤面がなくboards_ok=Falseで抑止。直前試合の非ゼロ得点80,221 / 29,336を
保持しており、gamesのclearで境界証拠は失われていない。実動画では2試合とも
history_complete=False。受理陽性は保存・live双方の統合テストで確認した。

| 指標 | 最終実走 |
| --- | ---: |
| 評価worker通知数 | 1,032 |
| 1通知 P50 / P95 / 最大 | 4.112 / 26.126 / 207.076 ms |
| RSS標本数 | 35 |
| RSS最初 / 最後 / 最大 | 534.883 / 549.824 / 549.824 MiB |
| 起動前空きメモリ | 19,547,811,840 bytes |
| worker nice / BLAS | 全標本10 / 各1 |

原票: `logs/rt_dv2h2b_20261003/RESULT.json`、`live/audit/`。
集計は `verified=True`。短区間の接続検証であり、全長の性能合格を意味しない。

再実行:

```bash
bash scripts/run_integration_dv2h2b_checks.sh
bash scripts/run_rt_dv2h2b_final.sh
bash scripts/run_rt_dv2h2b_live.sh
bash scripts/run_rt_integration_checks_20261002.sh -m scripts.summarize_rt_dv2h2b
```

logs配下の追跡画像は基点161件→統合後161件、集合一致。画像差分0。
監査原票はgzip圧縮版もコミットし、モデルや動画の複製は含めない。

最終検証完了: 2026-10-03 04:04:15 JST。
A/B全体の経過時間は開始03:21:43から42分32秒（最終コミット前）。
Bのマージ開始03:37:44からの経過は約26分31秒。
Aの最終状態は `4d9c654`（統合 `b3c5d1b`）、Bの主統合は `7e66a15`。
全リポジトリpytest・全18記録の採点再実行・長時間ライブの検証は行っていない。
