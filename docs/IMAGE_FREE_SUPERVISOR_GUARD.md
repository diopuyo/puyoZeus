# 監督の進捗読取競合対策（2026-10-09）

PC1の監督が05:53、`progress.exists()` と `read_text()` の間のDrvFs置換競合で FileNotFoundError となり、finallyで全子ワーカーを停止した。固定計算ソースは変更せず、`controls/supervisor.py` の起動時に `ResourceBudget.observe` を実行時差し替えする。ファイル欠落・JSON途中読取りは、その経路の当該巡回だけを飛ばす。正常に読める場合のRSS計測・起動判定は既存処理をそのまま使う。

tickの通常例外は記録して次巡へ継続する。明示終了の SystemExit/KeyboardInterrupt は捕捉せず、通常の終了処理を維持する。PuyoLive、空き6GiB、CPU予約、認識本体・モデル・キャッシュキー・全出力は変更しない。初期ソースSHA照合は従来どおり起動前に実行する。

`supervisor_guard.json` と `supervisor_guard_events.jsonl` に監督PID、実行時差し替え件数、読取スキップ・巡回例外件数、tracebackを記録する。`observe_replacements=1` が採用確認であり、0件・票なしを成功とみなさない。ログ自身の書込み失敗はstderrへ出し、二次例外でワーカーを停止しない。

関連10テスト合格。FileNotFoundError/JSONDecodeErrorの注入、他の経路の観測継続、非ゼロの差し替え件数、tick例外後もmainが子を保持して次巡へ進むこと、明示終了時だけ子を終了することを確認した。

PC1は08:38 JSTに `setsid -f` で再開、監督PID431824、同一の独立SID431824、起動shell終了後の親はWSLの/init。c68新140100、d01新131100、d04旧90900、d01旧前方7500の最新CPから継続。d04新99002は完了済みとして再起動しない。実測票は `b3_data/codex_tasks/image_free_speed_history_v1/supervisor_guard_v1/measurement.json` に保存する。

PC2は現在の比較を止めず、`share/tasks/inbox/bh_pc2_supervisor_guard.md` の手順で次の安全な機会にcontrolsだけ適用する。配布ファイルとSHAは同じ `supervisor_guard_v1/` に置く。PC2側の追加変更があれば保持する。pushしない。
