# リリース既定の変更（2026-11-01）

**11/2追補を反映し、24w2の設定を配布入口で読み込んだ実効設定との差は、利用者設定3項目と診断2旗だけになった。** 配布物の組立・更新・OBS実機試験は未実施。

## 採用した既定値

**A：以下19旗はすべて OFF → ON。** 根拠：PC1の11/1指示と全コマ確認済み24w2のモデル・学習列・表示の組。ensembleと直接公開は22:4x追補でDからAへ移した。

- `live_24h`・`live_24h_retr`・`live_24h_xc`・`live_24h_tape_next`
- `live_24h_media_fresh`・`live_24h_row_time`・`live_24h_retr_h_forecast`・`live_24h_early_submit`
- `live_24h_stable_key`・`live_24h_xc_same_time`・`live_chain_split_place`・`live_b3_fire_board`
- `b3_c2`・`b3_skip_icon_reading`・`live_24h_retr_second`・`live_24h_retr_move050`
- `live_24h_retr_deadline_extend`・`b3_ensemble`・`live_x_direct_publish`

**B：以下14旗はすべて OFF → ON。** 根拠：同指示による認識・終了・復帰・停止の修正採用。death_check_retrとverify_read_fastは11/1追加。

- `live_frame_drop_robust`・`live_lockdown_score_zero_then_move_release`・`live_match_end_pair`・`live_match_end_feed_on_hold`
- `match_end_yatta_text_template`・`live_capture_reconnect`・`b3_worker_graceful_stop`・`live_verify_soft_hold`
- `live_native_hsv_main`・`live_native_read_board`・`b3_share_score_read`・`live_color_roi`
- `live_death_check_retr`・`live_verify_read_fast`

**前提・運用値：**

- `b3_judgment`・`b3_async`・`b3_m3`：OFF → ON（Aの前提。常駐では前二つは従来から明示ON）。
- `live_24h_ex_speed`・`live_24h_ex_fire_first`：OFF → ON、`live_24h_ex_fresh_sec`：0.5秒 → 2.0秒（Aの24w2運用値）。ama=ON、遅延200 msは維持。
- `live_disable_power_throttling`：未指定時は環境変数、未設定ならOFF → 配布既定ON（10/30提案と11/1指示。窓なし起動の省電力抑制を解除）。明示OFFは優先する。
- `b3_model`：`models/b3/model.joblib` → `models/b3_c2_ens/ensemble_ref3_logit.json`。`live_24h_model`：未指定 → `models/b3_ex_L200H_retR/ensemble.json`。いずれもアプリ内の場所。timing_tableの場所は維持。
- PC2結果待ちの「(d) 連鎖の時間の表」は未採用。モデルの中身・SHAは変更していない。

**11/2追加採用：以下19旗も OFF → ON。** PC1の11/2 01:2x判断による。根拠は24j〜24w2のOBS試験と全コマの確認でONのまま回ったこと。旗と明示Falseの指定は残す。

- `b3_time_gate`・`live_score_cadence`・`live_static_exact`・`fast_first_judgment`
- `evaluation_freshness_gate`・`terminal_display_hold`・`live_disable_placement_reconcile`・`b3_score_matmul`
- `live_reference_times_only`・`b3_ledger_mode`・`b3_ledger_landing`・`live_near_future_cache`
- `live_exact_connectivity`・`live_exact_native`・`live_verify_async`・`live_board_vectorized`
- `live_hsv_lazy`・`live_recog_speed_py`・`live_recog_speed_py_pair_lazy`

同じ判断で、次の2点も採用した。

- `PUYO_LEDGER_PARTIAL`・`PUYO_X_BOARD_ONLY`・`PUYO_X_B3_BOUNDARY`：常駐の子プロセス環境とlauncherの起動準備で、未設定なら `1`。既存値は `0` や空文字を含めてそのまま優先する。OSの永続環境設定は変更しない。
- `dshow_color_correction`：機器名が `OBS Virtual Camera` の場合の配布既定を `off` → `601to709`。他の機器は `off` のまま。JSONの明示指定は優先する。

採用値は `production_config.py` の `RELEASE_LIVE_OBS_FLAGS` / `RELEASE_LIVE_ENV_DEFAULTS` / `RELEASE_OBS_*` に日付と根拠を記載した。診断2旗はOFFを維持する。

## 起動口と戻し方

利用者入口は常駐アプリ。機器・内部ポート・出力先・表示は従来の設定を使い、最小設定だけで上書きせず採用既定を含むJSONを生成する。既定は `src/production_config.py` の `RELEASE_LIVE_DEFAULTS` / `RELEASE_LIVE_MODEL_PATHS` 一か所で定義し、`release_config.py` を共用する。
旗は残し、JSONの明示値を優先する。前提条件の検証・24hによる旧表示の無効化・資産不足時のエラーは維持。低水準の `parse_user_config` / `UserConfig` と実験CLIは比較再生用の従来値を保つ。利用者用の `load_user_config` / `prepare` に配布既定が入る。auto_sequenceや既存の入口・組立台本は削除していない。

常駐のJSONは起動時に再生成するため、配布既定を恒久的に戻す場所は上記の単一情報源。開発用launcherのJSONでは明示値で比較できる。

## 削除と保持

**削除0件/候補11項目。** 下記すべてにsrc参照があり、未使用を証明できないため保持。native未搭載時の代替経路も変更なし。

- `live_x_prefire`・`live_board_vectorized`・`live_recog_speed_py`・`live_recog_speed_py_pair_lazy`
- `live_winrate_switch`・`live_hsv_lazy`・`live_v49f`・`live_ama_plain`
- `PUYO_LEDGER_PARTIAL`・`PUYO_X_BOARD_ONLY`・`PUYO_X_B3_BOUNDARY`

環境変数3項目は現基点の `ledger_x_options.py` に読取がある。56項目の原案は全文確認済み。各旗の検索件数と根拠はジョブの `flag_audit_56.json` / `old_references.json` に保存した。

## 24w2との差

**配布入口で読込後107項目中、一致102・差5。** 常駐生成JSONと24w2 JSONを実際の `load_user_config` で読み込む。差は `device_index`・`port`・`overlay` と `live_stage_timing`・`live_verify_reason_log`（ON → OFF）のみ。モデルパスは同じ仮資産の配置へ正規化する。推論やOBS実機の一致試験ではない。

省電力抑制は24w2 JSONに記載がないため、配布入口では両方に採用済みのONが入る。配布既定を補わない旧 `parse_user_config` で原JSONと比較すると、この省電力差も含む6項目となる。生JSONの差を隠さず別に検査・記録した。元のOBS実行時の省電力状態を今回の設定試験から推定しない。

原JSONは87キー、SHA256 `783a489b42a69dd1675e0c4fc02dcf88272ce7f2ce39473cda6c04c31d99623e`。原本の一致を再確認し、fixtureも同じSHAを検査する。環境変数3件はOBS台本の指定 `1` と配布の未設定時の値が一致する。既存環境がある場合は利用者指定を優先する。

## 検証

11/2追補の関連10ファイル329件中、328件通過・既存スキップ1件・失敗0件。追加ケースIDは58件（既存1件のパラメータ化を含み、純増57件）。基点に対する新しい失敗0件、新しいスキップ0件。19旗の両入口・パイプラインへの伝達とFalse優先、環境の既存値保持、機器別色補正、実効差5件と生JSON差6件を検査した。詳細はジョブのRESULT3に記録する。前回までの集計はRESULT2に保存し、無変更の全コマ再生・計算試験は再計算しない。

## 11/2 PC1の案・userの確認待ち

- live_chain_forecast_recheck・live_chain_end_formula: 配布の既定OFF→ON。連鎖の修正、13試合の確認で悪化0件/80対象。比較用の低水準既定はOFFを維持。
- live_verify_entry_fast_retry: 配布の既定OFF→ON。W64の入口の再試行を0.25秒にする。
- distribution_app_window: OFF→ON。userの注文による常駐窓。
- 表情切替ON・配置center・ポップ黒は既存の表示既定を維持。
- PC1 23:0x指示: 配布環境の既定にPUYO_OBS_AMA_RUST=1、PUYO_OBS_AMA_RUST_THREADS=2を追加。指定24w2台本に両変数の代入はなく、2は同梱コードの既定。
- 同台本のCPU_ISOLATION=0・AMA_IDLE=0・EXACT_SEED=1・PREPARE_EXACT=1（いずれもPUYO_OBS_接頭辞）も配布環境へ追加。FAST_AUDIT・DEEP_TIMINGは診断用のため追加しない。
- 参照exeがない場合は明示0でもRustを使い起動記録を残す。参照exeがある開発環境では0を守る。
