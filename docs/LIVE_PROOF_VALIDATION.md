# 単盤面native化の調査と検証

対象はbuild jの残存停止。C:のみで作業し、d〜jは書き換えない。

- `native/puyo_core`の既存bitboardはUNKNOWNを空として入力し、Pythonの固定壁と異なる。
- `src/puyo_core_bridge.py`の`simulate_chain`は既存nativeを使うが、全段の消去色・グループ・盤面を返さない。
- `NativeSimulator`は通常の個数特徴量の近未来探索で採用済み。死亡証明の`Search.simulator`は`ChainSimulator`。
- `Search.step`は楽観応手を計算し、`FastExpander`で配置を展開する。静止判定・局所連結・発火した子の連鎖計算がPythonに残る。
- 楽観応手は`ExchangeLandingProjection._optimistic_response`→`future_send`→`near_future_fire_power`。ここは`_FUTURE_SIMULATOR`（Pythonの`ChainSimulator`）を使い、個数特徴量の`NativeSimulator`経路とは別である。
- `exchange_prefire_candidates.placements`は既にnative。ただし`completion`は各段の得点・消去色が必要なためPythonへ戻る。
- `HiddenRowPrefire._options`は上記`completion`を2,798候補で呼ぶ。保存最大t=7027.0988605の主因。
- h frame20655はhidden_proofの48候補・Search.step 6,944回・近未来探索6,908回。原票は`<work>/stall/`。
- 新経路は独立Rust C ABIを`native/puyo_core/src/live_exact.rs`に置き、既存拡張を変更しない。単盤面の連結・消去・落下だけを実行する。
- `live_exact_native`は既定OFF。評価子の`bounded_chain_caches(future_search=True)`内だけ適用し、キャッシュ・候補順・探索上限・死亡証明・EMAは維持する。
- 全段の前後盤面、グループ順、隣接おじゃま、dtype、得点をPython結果型へ復元する。UNKNOWNは固定壁、隠し段のおじゃまは従来同様に隣接消去する。
- 入出力とビルド・再投入のreceiptは`<work>/proof/`。重い処理はHOLD不在・BelowNormal・空き6GiB監視・直列。
- 局所連結のDLL呼出しはhで4.16→4.88秒に悪化したため不採用。`h_on_local_ffi/`に除外試行を保存し、最終構成はjの局所判定を維持する。
- DLLの再生成は`native/puyo_core/build_live_exact.sh`。既設WSL Rust 1.97.1とZig 0.16.0を使用し、ビルド出力・一時領域・ZigキャッシュをC:に置く。既存`puyo_core`拡張は変更しない。
- OFF/ONはj相当の近未来キャッシュ・連結短縮を共通ONとし、新しい`live_exact_native`だけ切替。48,367通知の参照・表示・EMA・署名一致、FIX4の表示3,191行の全列dtype/バイト・events・死亡履歴/証明一致（従来の費用3列のみ除外）。
- t=7027.0988605の更新＋参照は1,249.718→754.288ms。保存全体の最大は1,249.718→1,007.067ms（ONの最大はt=6615.1283658）。
- h frame20655を含む348通知は全署名一致、停止通知4,158.351→3,827.061ms。gの同一試合548通知も全署名一致、4,153.477→3,827.049ms。
- 全保存入力の更新＋参照総時間236.254→190.548秒（19.3%短縮）、P99は90.257→34.799ms。未完了通知を新たに採用したり、通知を省略したりしない。
- 数秒の停止は残る。以上は同一保存入力のCPU計算費用であり、OBS実機での表示停止解消を証明するものではない。
- 関連227テスト合格。追加拡張したnative用18テストでは40,000ケース（隠し段ON/OFF・int8/uint8・Python通常/高速グループ）と候補順・ノード数・証明を照合して合格。
- 保存したビルドレシピで別出力先へ再生成したDLLも同じ18テスト・40,000ケースで合格。配布には全通知再投入済みのDLLを使用する。
- 配布先は`C:/PuyoLive_20261012k/PuyoLive`、device_index=3、kだけ`live_exact_native=True`。MANIFEST887/887、Git管理ファイル461件のSHA一致。
- 同梱Python13,124ファイルの追加・削除・変更0。従来非コード資産346件とモデル・パッケージは基準と同じ。追加DLLもMANIFESTで検査する。
- 同梱Pythonによる起動試験でPNG60枚・B3評価28回・保存参照185件を確認。新DLLのロードと評価スコープへの設定反映も確認。D:読取を遮断する監査で0件、起動設定のD参照0件。
- 重い処理は直列・BelowNormal・開始前HOLD確認・空き6GiB監視。最終検証と配布の成功receiptで空き最小17.252GiB。d〜jは変更せず、pushなし。
