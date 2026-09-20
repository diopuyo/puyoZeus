# 色変更と保存検証の合成候補

親所有: live_cli.py / test_runtime.py / preflight.py / run_cpu.py / launcher.sh / CONTRACT.md。
製造中。独立した新proof設計・CPU・実保存反例の検証が終わるまで実動画起動禁止。

認識候補はg2_palette_evidence_veto_2026-09-09_v1を無変更で再利用する。
旧失敗video38_palette_veto_live_2026-09-09_v1のexit1とCOMPLETE不在を保持する。
取得済み失敗原票による検査器のCPU試験は別rootへ出力し、失敗runの救済とはしない。

終了順序は色veto保存→旧journal背景の終了検査。色証拠を旧prefix検査より先に閉じる。
旧境界前の不変という主張は色介入と両立しないので、そのFalseを保持する。
非対象値/raw/NEXT/Counter/順序/時刻/元collector採録の回帰を別の実証拠で検査する。
一致フラグの無条件True化、例外の握りつぶし、未知差分の除外は禁止。

新proofインターフェースは独立診断後に確定する。現時点でquality gateや本番/学習権は発行しない。
