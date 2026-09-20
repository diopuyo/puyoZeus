# FIFO色束縛と最新palette合成のCPU契約

既定 OFF。`loaded_latest()` → `compose(latest, enabled=True)` → 元configuration内の `configured(latest, runtime, finisher, driver, addon, engine, enabled=True)` を使う。新CLI/GPU起動は提供しない。

原 pending builder・T2・FIFO・grace validator/compile を実行して二つのcodeを導出する。recognition本体やモデルfactoryは実行しない。OFF二codeが原journal固定値に一致することを必須とし、ONの全code構成も実install時に検査する。旧sourceを編集せず、別名journal moduleの期待値にだけ新二codeを設定する。

渡された導出dictは固定sourceからの再導出と型を区別したJSON完全一致を要求する。任意CODE_HASHESやenabled=1を自己申告して許可する入口ではない。保存/再読でも再導出を照合する。

最新T2 installの同呼出直後にFIFO installを追加。元palette/atomic addonと保存順序、原proofの条件は維持する。proofが再読用journalをロードする経路にも同一導出の私有viewを渡す。原proof.load自体は呼び、scope終了後に復元する。

`FIFO_PALETTE_COMPOSITION.json` は新出力へ排他保存。導出元source SHA、OFF/ON code、原C6/grace receiptを保存する。旧runへ追記しない。原journalは旧固定CODE_HASHESのままで、未知code拒否を維持する。

CPU合成の合格は物理完了、G2、会計、品質、本番許可ではない。FIFOで認識結果が変わる場合、既存proofが持つ全保存比較条件に引き続き従う。finish/全実保存の通過をこのcompile-only試験から主張しない。
