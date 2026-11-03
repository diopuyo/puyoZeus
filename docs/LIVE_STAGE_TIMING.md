# live 段別時間 (2026-10-12)

`live_stage_timing` は既定OFF。配布f設定でのみONにする。判定通知と表示DTOには列を加えない。
出力は実行記録の `spool/frame_timing_*.jsonl`。JSONLを逐次保存し、RAMへ全履歴を保持しない。

- `recognition`: 入力ごとに取り込み、正規化、色補正、画面確認、連続性、盤面認識、B3生得点・式・会計・予告・観測器・キュー、認識通知送信。
- `evaluation`: 受信frameごとの評価全体の壁時計・thread CPU時間。
- `event_evaluation` (2026-10-13追加): 子評価の通知復号・状態/評価更新・物理時刻参照・返信。親側の `event_ipc_send` / `event_ipc_wait` と request_id および入力frameで照合する。子CPU時間は親CPU時間に含まれない。
- `b3_result`: 計算した元frameのB3特徴生成・探索・推論・待ち時間。非同期では受信frameと異なり得る。採否と世代も記録。
- `publication`: 公開threadのDTO生成・検証・SSEキュー投入。ブラウザ受信時間は既存の送信監査を使う。

単位は `*_ms` がms、`*_at` と `t_sec` が秒。壁時計は `perf_counter`、CPUは所有threadの `thread_time`。
`stages` の `calls=0` は未実行。未認識の入力保留行は `frame=null`、`held=true`。結合キーはframe単独でなく `captured_at` と入力世代も参照する。
盤面認識の時間には DeviceSession の較正管理を含む。段の合計に入らない通知凍結・監査・Python処理は全体との差で調べる。
非同期画面確認の `screen_verify` は所有者での投入・回収時間。実検証時間は `details.screen_result.duration_sec`、対象画像は同じ辞書の `captured_at`。
非同期画面確認は継続検証中に前回合格を維持するため、不合格の反映は完了後の入力処理時。入力断・形式変更は直ちに未検証へ戻し、旧結果を破棄する。
初回・保留中は認識が未開始なので、workerで同じ画像の検証を完了してから認識へ渡す。入口で検証待ちの画像を読み捨てない。
時間の差にはOS待機とGIL待機の両方を含む。CPU時間との差だけで原因を断定しない。
短い段のCPU値は時計の分解能の影響を受けるため、複数フレームを集計して読む。
旧eの `evaluation_stages.pickle` は子プロセス内を計装していない。段が0でも未実行を意味せず、子評価とIPC待ちは全体との差に含まれる。旧データに新しい計時列があったものとして解釈しない。
外枠猶予ONでは `details.screen_result.diagnostics` に通常検証・猶予・辺ごとの画素差を記録する。期限は通常合格から3秒、猶予自身では更新しない。
