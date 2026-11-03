build_20261012kを作成。live_exact_nativeは通常既定OFF、k限定ON。diagnostic_not_quality_pass。
起動: C:/PuyoLive_20261012k/PuyoLive/PuyoLive.bat（device_index=3）。
残存Python経路: 死亡証明のSearch.step→future_send→FastExpander、HiddenRowPrefire._options→completion。
既存RustはUNKNOWNを空扱いするため変更せず、UNKNOWN壁を維持する単盤面Rust DLLを追加。
評価子だけで静止判定・連鎖の連結/消去/落下をnative化し、各段の完全なPython結果型へ復元。
候補・探索順・ノード上限・通知・死亡判定・EMAを維持。打切り追加・間引き・GPU・バッチ化なし。
局所連結のDLL化はhで4.16→4.88秒に悪化したため不採用。最終版はjの局所判定を維持。
関連227テストと拡張native18テスト合格。乱数40,000条件で全段・得点・dtype・終端・入力不変を照合。
隠し段ON/OFF・おじゃま・UNKNOWN・int8/uint8・通常/高速Pythonを含む。再生成DLLも同じ照合に合格。
OFF/ONの全48,367通知で参照・表示・EMA・署名一致。FIX4基準で表示3,191行の全列dtype/バイト一致。
死亡履歴・証明・eventsも一致（従来の費用3列のみ除外）。j相当のキャッシュ・連結短縮は両側ON。
h frame20655区間348通知: 4,158.351→3,827.061ms。g区間548通知: 4,153.477→3,827.049ms。各全署名一致。
t=7027.0988605: 1,249.718→754.288ms。保存全体最大は1,249.718→1,007.067ms（ON最大は別通知）。
更新+参照の総時間236.254→190.548秒（19.3%短縮）、P99 90.257→34.799ms。
MANIFEST887/887・Git管理461ファイルSHA一致。同梱Python13,124ファイルの追加/削除/変更0、モデル・従来資産346件・依存は基準と同じ。
同梱起動試験: PNG60枚、B3評価28回、保存参照185件一致。新DLLロード・設定反映確認、D読取遮断監査0件、起動設定D参照0件。
全重処理は直列・BelowNormal・開始前HOLD確認・空き6GiB監視。d〜j変更なし、pushなし。数秒の停止は残り、OBS実機品質合格ではない。
原票・比較・入力SHA・配布receipt: <work>/proof/。詳細: wt_live_speed/docs/LIVE_PROOF_VALIDATION.md。
コミット: 診断8f7f487d、実装7c6af0b4、採用89caaffc。採用情報の単一情報源はsrc/production_config.py。
