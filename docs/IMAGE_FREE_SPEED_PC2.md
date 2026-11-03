# PC2 高速化準備版（段3未合格・起動禁止）

区間分割は3か所の旧経路照合で不一致となったため使用しない。
この一式はキャッシュ・自動並列数の配布準備版であり、修正候補の検収済み配布物ではない。
`quality_gate.json` は未合格を記録しており、起動入口は全区間比較を拒否する。
モデルは前回の比較用のまま。fix1 完了後、統合したソースで段3を検収して最終版を再生成する。
未合格票を手作業で合格へ書き換えない。

1. 共有 `\\192.168.1.2\puyo_d\b3_data\codex_tasks\image_free_speed_pc2_v1.tar` と
   `.tar.sha256` をPC2のローカルへコピーし、`sha256sum -c` で照合する。
2. 新しいディレクトリへ `tar -xf image_free_speed_pc2_v1.tar` で展開する。
   既存の `image_free_parallel_pc2`・`image_free_recog2` は上書きしない。
3. 依存環境は前回同様Python 3.12。`requirements-pc1.txt`、`requirements-pc2.txt`、
   `wheels/` を使い、WSLから `powershell.exe -NoProfile -Command Get-Date` が通ることを確認する。
4. **PC2では動画記録 (`al_`) と研究測定 (`ax_`) の並列を絞り、比較にCPU・メモリを最優先で回す。
   user指示: 画像なし認識が最優先。** PuyoLive検出時の停止は維持する。
5. 最終版の合格後のみ、`PUYO_PYTHON=/絶対パス/venv/bin/python` を設定して、
   `setsid -f bash run_pc2.sh >> parallel.log 2>&1 < /dev/null` で起動する。
   この準備版では段3未合格エラーで終了する。PC2は今回まだ起動しない。
6. 最初は1ワーカー。300フレーム経過後のRSSとCPU負荷を実測して並列数を決める。
   Windows・WSL両方に6GiBを残す。現方式は動画数が上限で、PC2担当d01/d02は最大2。
   `supervisor.json` の `measured_peak_rss`・`effective_workers` と進捗で実測を報告する。
7. キャッシュは旧経路のコード・モデル/旧画像・動画の内容SHAと依存版をキーとする。
   300フレームごとに旧出力と終端状態を保存し、再利用時は先頭8フレームを実計算する。
   不一致ならそのチャンクを捨てて計算する。新経路の画像読込禁止は維持する。
8. 完了後は新しい `image_free_speed_pc2_results_v1` へ結果・差分・チェックポイント・
   ソース/モデルSHA・キャッシュ検証票を戻す。PC1の既存結果へ混ぜない。

分割未採用なので、速度比例の区間割当と区間接合は実施しない。動画単位の場面数定義は従来どおり。
GPUはCPUとの全項目完全一致を未検証のため使用しない。PC2での実並列数・実測速度は未測定。
最終版を作る際はfix1の `death_0/death_1` 記録を保持し、段3合格ソースSHAと一致させる。
