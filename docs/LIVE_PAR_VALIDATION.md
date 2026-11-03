# live認識の並列候補（2026-10-13、未採用）

基準はmの `dc24ee44`。既存配布d〜m、他レーンのソース、Dドライブは変更しない。
配布pは作らない。指定の全件品質門には不足があり、速度だけで合格としない。

## 実装

- `live_onnx_parallel`：既定False。ONNXのintraを2、interを1のまま、待機スピンを無効にする。
  入力、前処理、モデル、softmax、判定閾値を変更しない。ON/OFFでセッションcacheを分離する。
- `live_color_buffer`：既定False。入力源ごとに最大3個の出力配列を持つ。
  CPythonで外部参照がない所有配列だけ再利用する。viewや非同期処理が保持中なら独立の出力を作る。
  同じ入力源の所有スレッドだけが補正を呼ぶ。画像保持先へ上書き許可を要求しない。
- 両フラグはproduction_configが既定値の単一情報源。launcherのJSON検証、CLI、spawn環境へ配送する。
  公開確率、通知順、時刻、評価キュー、CNN内部・差分認識は変更しない。

## 比較の定義と再現

`scripts/live_parallel_matrix.py` が候補を直列実行し、各試行を別processで初期化する。
最初の12条件は順方向→逆方向の24試行。追加は候補ごとのA→B→B→A。
PNGは既存framesAの210枚、先頭60枚を暖機し、既存台の規則で終端1枚も速度集計から除く（n=149）。
認識平均/P95は `live_process.recognize` 全体であり、画像だけを読む関数の時間ではない。
logits入力hash採取・出力保存の計装を含む。fpsはこの再投入台の処理周期で、実機fpsではない。
元動画はC上のvideo_d04、650〜680秒の900枚を無加工BGRで同一デコーダから投入する。
映像時刻は固定、先頭60枚・終端1枚を除く速度母数は839。原映像の再投入であり、実機の保存画素ではない。
元動画の基準側はGitから展開したmソース、候補側は実装済みフラグを使用する。

`scripts/compare_live_parallel.py` は全コマのresult_bytes、pipeline、b3、frame、t_sec、dropped_beforeを照合する。
壁時計captured_at/recognized_atは速度試験で変わるため品質比較から分ける。映像時刻は除外しない。
logitsはモデル名＋入力batchのSHAで対応付け、全出力bytes、argmax、最大絶対差を照合する。
初回の既存bench digestは広い名前規則で時刻を除外するため、合格根拠は上記の明示フィールド比較を使う。
raw_cnn_board・cnn_board・confirmed_boardはそれぞれセル数と欠測不一致も記録する。
同じ映像の繰返しは独立した新しい品質標本として数えない。

## GILと左右分割

NumPyは多くのネイティブ処理でGILを解放するが、全操作が対象ではない。
小さな配列の処理、Pythonのセルループ、リスト・Board構築は別途費用が残る。
[NumPy公式のスレッド安全性](https://numpy.org/doc/stable/reference/thread_safety.html)。
OpenCV 4.13.0の生成ラッパー用ERRWRAP2はPyAllowThreadsでC++呼出を囲む。
[公式実装](https://github.com/opencv/opencv/blob/4.13.0/modules/python/src2/cv2_util.hpp)。
ONNX Runtime 1.30.0のrunは入力変換後、Session::Runの区間でgil_scoped_releaseを使い、Python出力組立前に戻す。
[公式実装](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/python/onnxruntime_pybind_state.cc)。
実環境はPython3.12.3 / NumPy2.4.4 / OpenCV4.13.0 / ORT1.30.0。

左右のread_boardをThreadPoolExecutorで重ねた診断ではCPUが約1.1コアに留まり、処理時間は悪化した。
read_boardは側別彩度較正時に共有分類器のスケールを一時変更するため、診断でもその設定を明示拒否した。
共有較正を無視した製品スレッド化はしていない。プロセス分割は未実装・未測定。
ONNXのスレッド増加だけでは、2つのモデルの待機スピンが合算され、最大約7コアを消費した。
追加したスピンなし条件と区別する。CPU使用量が増えたこと自体を高速化と呼ばない。

## 準備段と入力保持

準備段の診断は固定画像のcopy（capture相当）＋色補正を次の1枚だけ先行し、現在の認識と重ねる。
画像順と映像時刻は固定し、得点・式OCR、状態遷移、b3_queuesは従来の認識所有スレッドで順番に処理する。
本物のDirectShow取得時刻や画面検証のタイミングが同一になることは証明していないため、製品へ配線しない。
バッファ候補は画像全体/ROI view/非同期保持・入力非破壊・形状/dtype切替をテストした。
全210画像の補正出力（各6,220,800bytes）を直接比較し、連結SHAも一致した（color_pixels.json）。
画像静止ゲートやHSVの再読を省略するb3_queues変更はしていない。

## 指定保存入力の制約

g=76,800、h=82,749、j=83,425、k=82,903、e=48,367更新（計374,244）。run_iは不在。
全5ファイルにcomplete行なし。読取例外・SHA・種類別行数はinput_inventory.jsonへ記録した。
ExchangeEventRecorderは認識後の評価境界を保存し、画像・logitsを保存しない。
したがって、この374,244通知を評価器へ再投入して一致しても認識変更の全件検証の代用にはならない。
過去runと同一画素による全件認識・評価行・公開確率の一致は未証明。

## 負荷・長時間試験

各開始前HOLD確認、認識processはBelowNormal、各コマ境界で空き6GiB下限を確認する。
優先度比較はBelowNormal対Idleのみ。高優先度化して他レーンやOBSを圧迫しない。
CPU set比較は性能側12個対4個の論理CPU集合。物理4コアとは呼ばない。
各試行の全体CPU平均/P95、認識processのCPU秒/秒、開始終了時刻・空きメモリを原票へ残す。
初期探索と追加候補試験中には関連テスト・入力棚卸しも実行した。最終900コマ対はこれらと重ねない。
OBS相当の人工CPU負荷は未実施。他レーンの同居負荷をOBSの再現とは扱わない。

55分試験はdevice_index=2、port=18843、OBSなし、内部m→候補の直列対を用意する。
先行deltaとqueueの予約完了、および60秒連続のカメラ解放を待つ。先行失敗時は追い越さず停止する。
途中でHOLD/6GiB未満/別feedを検出した場合は所有試験だけ停止する。状態はcamera_status.json。
内部版をMANIFEST照合・同梱Pythonのcheck-onlyで確認してから起動する。配布pとは別物。
内部mは889/889、候補は893/893。さらに各同梱Pythonで60枚の認識と28回のB3評価を実行した。
両方とも終了コード0、起動監査のD参照0。B3生確率・映像時刻・frame・triggerは28行一致した。
この生確率の短い一致を公開確率や全評価行の検証とは呼ばない。
capture→評価終了のP50/P95/P99/最大、公開遅延、CPUコア数、55分の劣化は試験完了まで未測定。

## テストと未合格事項

直接関連する単体試験は156合格・ONNX書出し依存による2skip。
広く追加実行したlive_b10/b12は83合格・8失敗。失敗はWindows非対応のos.getloadavg/sched_setaffinityと
同梱Pythonにないimageio_ffmpegであり、今回変更外。これを全テスト合格とは記録しない。
実際のONNX推論は別の画素再投入試験で実行するため、書出し試験のskipを推論一致の代用にしない。
生記録は `logs/live_parallel/`。小さい集計と一致結果は `docs/live_parallel_evidence/`。
短い結論は `<work>/LIVE_PAR.md`。過去画像がない品質門の代替条件にはユーザー判断が必要。
