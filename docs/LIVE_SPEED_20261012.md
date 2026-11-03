配信認識高速化・C:単独OBS試験用配布版（2026-10-12 10:17）

- 配布先: C:/PuyoLive_20261012d/PuyoLive。PuyoLive.batで起動、設定はpuyo_live.json（port 8765）。
- ブランチ: claude/obs-live-speed-20261012。実装6d0f650f、5秒merge 8ff4261b、配布コード089db251。pushなし。
- 高速化3フラグはproduction_config.pyで既定OFF・採用日2026-10-12と根拠を登録し、この配布設定だけON。
- live_disable_placement_reconcile=true: デバイス配信のみR1b停止。動画30fpsは維持。再有効化は安定25fps以上＋合図の有効性を再検証。
- b3_score_matmul=true: B3得点OCRに既存matmul経路を適用。
- b3_warning_parallel=true / b3_warning_workers=4: 単体CNNを並列化、LRU排他・読取専用推論・worker回収を実装。
- 決着保持: claude/obs-terminal-hold-20261012の8e29ad2aをmerge済み。上限5秒（次試合・機器切替で解除）。
- D退避: 09:44:38完了。build bのapp/参照設定・native/ONNX wheel計876ファイル75,320,157 bytes。assets/D_INPUT_SHA256.jsonでコピー前後SHA一致。以後D読取なし。
- 決定digest: A 614/614・B 600/600（R1b停止含む）、C 900/900（R1b維持・B3高速化のみ）。通知全体＋B3 payload、合計2,114件すべて一致。
- OFF互換: 同じA/B/C合計2,114件も元build bと全一致。
- core認識 P50/P95: 151.6/173.5 → 59.9/77.2 ms。旧→新→旧→新、各210枚・先頭60枚除外、各条件298測定点、2回の百分位の中央値。
- full取り込み周期 P50/P95: 149.2/228.7 → 61.4/116.5 ms（P50約16.3fps相当）。同じ交互条件、PNG再生による測定。
- CPU（認識process）: core 0.97→1.79、full 0.97→1.75コア。cpu_threads=1維持。BelowNormal・直列、速度試験の空きメモリ最小9.48GiB。
- 単体・関連既存: 218 passed / 1既存skip。assets/tests_final.xml。
- 配布照合: MANIFEST 872/872一致、git管理453ファイル一致。Python元はC:/PuyoLive_20261012a/PuyoLive/python、13,124件コピーSHA一致。最終差分は再生成pyc 9件のみ。
- 起動確認: 同梱PythonでPNG90枚・B3評価40回成功。D読取を遮断しPATHをC限定、読取等3,594パス・ロード済みDLL等287件にDなし。カメラ読取・ポート待受を禁止して確認。
- 文字列監査: 配布テキスト6,073ファイル、D文字列47件は説明/導入元/未使用収集既定値等、起動設定0件。証拠は<work>/assets/{delivery_receipt.json,bundle_static_audit.json,smoke/,speed/}。
- 残課題: OBS実機・仮想カメラ試験はClaude担当で未実施、18813不使用。OCR浮動小数点の未観測閾値境界とOBS同居時のCPU負荷は未保証。
