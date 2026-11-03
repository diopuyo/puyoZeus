# 配信の速度門：低スペック係数の実測台（2026-10-22）

**準備完了・新しい実測は未実施。** PC1のv4.12/M4走行を妨げず、配布21c・本番ソース・既定設定は変更していない。出力先はD:。旧「工程合計×1.647、500ms超0件」の門は今回の判定に使わない。

## 1. 1.647の出典

算出式は **max(176.5/108.3, 3263.8/1981.2) = 1.6473854229759741**。再帰探索のP50比とP95比の大きい方で、P95比が採られた。

- 式そのもの：[<work>/EXCH_REVIEW_RENDER.md:99](<work>/EXCH_REVIEW_RENDER.md:99)、同102行。追補4(c)「quick後の再帰」の比を、後の予測計算へ継承している。
- 通常側 **108.3 / 1981.2ms**：[AMA_RUST_F.md:625](<work>/AMA_RUST_F.md:625)。低スペック側 **176.5 / 3263.8ms**：[同693行](<work>/AMA_RUST_F.md:693)。これは前景の時間を除いた裏の再帰時間。
- 条件：[同571行以降](<work>/AMA_RUST_F.md:571)。boards36.jsonから3件おきの12盤面×3回＝36標本、両向きで1呼出、構成ごとの暖機1回を除外、線形補間。K12・幅4・深さ6・再帰3段、手時間733ms。映像入力・認識・HTTP公開を含む実測ではない。
- 通常は追加負荷・affinity制限なし。低スペック側は **0xF000（論理CPU 12〜15）＋同じコア群でCPU負荷1プロセス**：[同574行](<work>/AMA_RUST_F.md:574)。PC1ではこの4個はEfficiencyClass 0のEコア。周波数を電源設定で制限する操作やBelowNormal指定は、この計測器にはない。
- 実装：[bench_ama_addendum4.py:164](<work>/wt_ama_rust/scripts/bench_ama_addendum4.py:164)（入力抽出・反復）、[同199行](<work>/wt_ama_rust/scripts/bench_ama_addendum4.py:199)（affinity・負荷起動）、[bench_ama_rust.py:20](<work>/wt_ama_rust/scripts/bench_ama_rust.py:20)と[同61行](<work>/wt_ama_rust/scripts/bench_ama_rust.py:61)（マスク・負荷ループ）。元の実行順は[run_ama_addendum4.sh:11](<work>/wt_ama_rust/scripts/run_ama_addendum4.sh:11)の通常→低スペックで、ABBAではなかった。
- 原票：[addendum4/summary.json:1](<work>/wt_ama_rust/docs/benchmarks/ama_rust_f/addendum4/summary.json:1) の `free/lowspec.summary[config=c_background].background`。丸め前はP95 **1981.1704249950708 / 3263.8270750030642ms**、比は **1.6474236813883312**。旧定数は原票の丸め前数値ではなく、報告書の小数1桁の数値から計算された。`free.json:1`の実affinityは65535、`lowspec.json:1`は本体・負荷とも61440。
- 継承先：[src/exchange_count_prediction.py:12](<work>/wt_live_v49f/src/exchange_count_prediction.py:12)、[src/prefire_worker.py:15](<work>/wt_live_v49f/src/prefire_worker.py:15)、[packaging/summarize_live_obs_speed.py:11](<work>/wt_live_v49f/packaging/summarize_live_obs_speed.py:11)。今回は変更していない。

従って、旧係数を「同じPコア上で周波数だけ落とした比」や「配信全体の遅延比」と解釈できない。10/21報告のコア種類についての推定より、上記の元原票を優先する。出典の抜粋・SHAは `D:/puyo_analyzer/packaging/live_gate_coef_20261022/preparation/provenance.json`。

## 2. 今回の低スペック条件案

**通常A：21cの配布設定、全論理CPU、起動親Normal、開始前の電源プラン。低スペックB：性能側の4物理コア（HT兄弟を含む）に測定プロセス木を限定し、複製した電源プランでAC時のブーストを無効。** 最小プロセッサ状態5%、最大100%。基準クロック付近を使う条件で、GHz値を実測前に決めつけない。

PC1の読取確認では、Aのマスクは65535、Bは255＝CPU 0〜7（物理コア0/2/4/6、EfficiencyClass 1）。CPU番号の固定書き込みではなく、WindowsのCPU topologyから4物理コアを選ぶ。Eコア12〜15や起動親BelowNormalへ落ちない。配布版内部の認識CPU set・評価workerの優先度設定は両条件とも維持する。

コア数も絞る理由は、旧条件が4物理コアだったことと、21cには認識・B3・v4.9f・amaの並行計算があること。全コアのまま周波数だけ落とすと、4コア機で起きるCPU競合を取り逃す。4コア8スレッド・ブーストなしを今回の明示的な模擬条件とし、旧Eコア＋人工負荷の倍率とは区別する。今回は認識を含む実アプリを動かすので、旧条件の代理CPU負荷をさらに足さない。

Bで対象CPUのProcessor Performance P95が110%を超えた場合は、ブースト無効を確認できないとして正式係数を出さない。実機の低スペックPC全般を保証する係数ではなく、ここで固定した条件への換算値を作る。

## 3. 入口と走行手順

入口は1個：[Live_Gate_Coef_20261022.cmd](<work>/wt_live_v49f/packaging/Live_Gate_Coef_20261022.cmd)。場所は `<work>/wt_live_v49f/packaging/Live_Gate_Coef_20261022.cmd`。

1. 引数なし・`--help`は説明のみ。`CHECK`は設定SHA・CPU構成・動画コンテナの長さを読むだけで、動画再生・電源変更・測定を始めない。
2. PC1が空いてから、`Live_Gate_Coef_20261022.cmd RUN-ABBA-21C` を実行し、`PC1-IDLE-FULL-ABBA` を手入力する。PC1 Claudeによる非対話起動は **両方の明示語**を渡す：`Live_Gate_Coef_20261022.cmd RUN-ABBA-21C --confirm PC1-IDLE-FULL-ABBA`。片方だけでは始まらない。
3. 全4試行を自動実行する。AC接続・空きRAM6.5GiB以上・C:10GiB以上・D:30GiB以上・同居計算なしを開始前に要求。同居計算があれば、その計算を止めず入口を失敗終了する。途中で計算が再開した場合も、従来の静穏監視で測定側だけを終了する。
4. 完了後、出力フォルダの `power_restore.json` の `restored: true` と `original_settings_unchanged: true`、各試行の `flow.json`、最後の `coefficient.json` を読む。途中失敗なら係数ファイルは作らない。

長時間の非対話起動は、上記cmdを `Start-Process -WindowStyle Hidden` で別プロセスにし、標準出力・エラーをD:へ保存する。今この手順を起動する必要はない。二重起動は拒否する。

**所要時間：全長3300.0667秒（55分0.067秒）×4＝3時間40分0.27秒。** 各試行60秒の切替待機、配布台帳照合・初期化・終了処理を含め約4時間を見込む。起動待ちの上限は各30分で、そこまで待つ場合は全体約5時間45分。P50/P95だけ20分に切り出し、走行そのものは途中で切らない。

## 4. 固定する入力・採取・集計

- 全試行で `build_20261021c_stage/PuyoLive` のコード・`puyo_live.json` を使用。設定SHA `1eec38bf16c3effe20cf69b1cdd266334522fc74495b472c3942d93f520fe7f1` と配布MANIFEST SHAを固定検査。ソースの重ね合わせはしない。入力を動画にするためのsource/video_path/start/end、監査用port/output/model絶対パスだけを外部設定に書き、元の設定ファイルは変更しない。
- 同じ `C:/puyo1_tools/soak/media/video_d04.mp4` を先頭から末尾まで、**A1通常→B1低→B2低→A2通常**。99,002フレーム・30fps。開始前に動画SHA `db5764831bb023fe090fda4eaed2fbf9e0aaade85efbd27256c13af6539d555a` を照合する。
- 21cと同じ監査・完全一致高速化・準備・詳細計時の設定を使用。HTTPのsourceフレーム/時刻が全長を進み、評価値も変化し、各20分区間に3枠が揃うことを `flow.json` に記録する。実時間で全長が経過したことも検査。例外・worker異常・不完全終了は失敗。
- 実遅延は、[live_publish.py:411](<work>/wt_live_v49f/src/phase_j/live_publish.py:411) のHTTP状態公開後に採る監査monotonicから、同じ入力のcaptured_atを引く。最新認識への時刻付け替えで古い評価を新しく見せないよう、**B3評価の元入力→公開を主値**、**最新取得→公開を併記**する。後者には死亡確定等の公開も含む。動画のcaptured_atは予定取得時刻で、復号・リサイズ・待ちを含む。HTTP受信・ブラウザ描画は含まない。
- **P50/P95：先頭0〜1200秒、暖機除外なし。P99/最大：全長。** 同じ入力の最初の公開だけを数え、母数nと500ms超件数・割合を記録。各20分ブロックと前半/後半で悪化を示す。全長のアプリ木RSS・プロセス別ピーク・前後半の増分・最小空きメモリも集計する。
- CPU周波数、Processor Performance、CPU使用率、電源プラン、同居プロセス負荷、メモリを約10秒ごとに全長記録。実プロセスのaffinity・優先度・メモリは配布版の約5秒監査でも照合。環境記録の欠落・途中の電源プラン変更・バッテリー駆動は正式集計を拒否する。
- 係数は各分位で **低÷通常**。条件別2試行をまとめたP50比・P95比・P99比に加え、B1/A1とB2/A2も出す。P95比で通常側の各実遅延を換算したP95・500ms超割合は参考欄へ出す。**割合の合格線はnullのまま**とし、実測後にuserへ諮る。今回、新係数や割合の線を本番へ採用しない。

出力：`D:/puyo_analyzer/packaging/live_gate_coef_20261022/run_日時/`。`protocol.json`、`environment.jsonl`、各試行の`polls.jsonl`・`output/b3_time_gate.jsonl`・`output/b3_resources.jsonl`、`flow.json`、`coefficient.json`、電源適用/復元票を保存する。大きな出力とTEMP/TMPはD:。

## 5. 復元と今回の確認

元の電源プランは変更せず、複製にだけ低条件を設定する。独立監視の準備と、次回ログオン用RunOnceの復元登録が済むまで電源を変更しない。通常終了・例外・Ctrl+Cはfinallyで監視へ復元を依頼し、controllerが強制終了されても独立監視が復元する。Job Objectの唯一の保持者をcontrollerにし、controller消失時は測定の子孫だけを終了させる。

復元時は元GUIDを再選択し、開始前の `/qh` と設定全文が一致したことを確認してから複製を削除する。電源断・再起動では次回ログオンのRunOnceが復元し、次のcmd起動も未復元票を先に処理する。手動再試行も同じ入口の `RESTORE`。復元失敗票は消さず、正式係数を作らない。

今回確認したのは、入口の説明/CHECK、門番・復元・集計窓の8単体検査、構文、保存済み21c B1原票の再集計だけ。新集計は既報のP50/P95・標本数・500ms超件数と一致し、全長確認・メモリ原票読み込みも通過した。**動画走行、電源設定の変更、実CPU制限付きABBAは実施していない。** 検査票は `preparation/` に保存する。pushなし。
