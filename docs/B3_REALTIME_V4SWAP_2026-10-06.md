# B3探索差替えとOBS試験の準備

非同期案の実装・600秒一致結果・OFF/同期ON/非同期ONの測定手順は
[B3_REALTIME_ASYNC_2026-10-06.md](B3_REALTIME_ASYNC_2026-10-06.md)を参照。
以下の「分離案（未実装）」はこの探索差替え時点の記録。

## 取り込み

基点は `eb913340a01b1e7ad6f000554b3e5683769814b4`。
前回と同じく、`wt_b3` の作業中ファイルではなくGitオブジェクトをファイル単位で取得した。
比較元は `d03a7da690eb5fa100abebba84294ea0a2f8153f`、取得元は
`c4ebcfb5f6f86070f7ed8944bf7b5cec8fda19f1`。

取得対象は `src/b3_{column_spec,features,search,timeaxis}.py`、`src/puyo_core_bridge.py`、
`scripts/b3_generate_rows.py`、`native/puyo_core/src/{b3,lib,bitboard}.rs`、
`tests/test_b3_{hidden_search,pair_search,timing_search}.py`、対応する列仕様・探索仕様・事前登録文書。
時間ベンチ・独立評価スクリプトと他課題の文書は取り込んでいない。
`bitboard.rs` の変更はB3専用シミュレータだけに適用し、既存指標の隠し段処理は維持する。

版は `b3_pair_potential_v2`、1手=2個組、撃たない候補を保持、連鎖後窒息判定、
隠し段おじゃま非消去。幅4・予備選抜倍率12・上限20組。
追加メタ列 `search_version` は末尾、対称化で保持、モデル特徴から除外する。
liveでは探索仕様を `b3_config.json.search` に記録し、旧nativeは入力開始前にエラーにする。
B3は既定OFF、`production_config.py` の採用設定は変更していない。
取得15ファイルは元SHAのGit blobと全件一致。証跡は `OUT/import_manifest.json`。

## 特徴の一致

証跡ルートを以下では `OUT = D:/puyo_analyzer/b3_data/codex_tasks/b3_realtime_v4swap` とする。
WSLの既存venv・c4ebcfbビルド済みnativeで、前回と同じ観測記録をlive入力口へ再生した。

```text
python -m scripts.b3_realtime_check --root D:/puyo_analyzer/b3_eval_records --rows D:/puyo_analyzer/b3_data/eval_rows_v5 --video b3_zenchi_s1 --max-frame 36000 --model D:/puyo_analyzer/share/results/pc2/final_v3/b3_final_v3_4e7b4da.joblib --timing-table D:/puyo_analyzer/b3_data/timing_v1/timing_table.json --no-timing --out D:/puyo_analyzer/b3_data/codex_tasks/b3_realtime_v4swap/parity_600s.json
```

WSLではパスを `/mnt/d/...` に読み替える。3,020行、欠落0・余剰0・共通列の差0。
参照側だけの `t_source_sec`、`winner_unknown` は後付け評価メタデータ。
v3の一括予測・live単行予測も保存行からの予測とビット一致、最大差0。
`--no-timing` は時計の読取りと時間集計を無効化する。今回は新たな性能測定をしていない。
観測記録からの一致であり、動画画素の再認識一致やv3の新特徴に対する予測精度の保証ではない。

## Windows配布Python用native

同じRustソースから `cp312-cp312-win_amd64` wheelを作成できた。
Windows上のmaturinは依存クレートの生成build scriptがOSのアプリ制御で拒否された
（`os error 4551`、`OUT/windows_build.log`）。OS設定は変更していない。
WSLのmaturin + Rust windows-gnu target + zigでクロスビルドした。

再現手順は `OUT/cross_build.sh`。事前に `native/puyo_core` を `OUT/puyo_core` へコピーし、
コピー側のCargo.tomlだけPyO3に `generate-import-lib` を追加する。リポジトリ側は変更しない。
既存のWSL zig環境 `<home>/zigvenv` を使用する。

```bash
export PATH=<home>/.cargo/bin:<home>/zigvenv/bin:$PATH
export PYTHONPATH=<home>/zigvenv/lib/python3.12/site-packages
rustup target add x86_64-pc-windows-gnu
cd /mnt/d/puyo_analyzer/b3_data/codex_tasks/b3_realtime_v4swap/puyo_core
export CARGO_TARGET_DIR=<home>/b3_rt_windows_target
<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -m maturin build --release --target x86_64-pc-windows-gnu --zig -i python3.12 -o ../wheels
```

出力は `OUT/wheels/puyo_core-0.1.0-cp312-cp312-win_amd64.whl`。
既存の `packaging/build/PuyoLive/python/python.exe` で別ディレクトリへ展開して読み込み、
12探索ケースのWSL一致と `LiveB3` 実初期化を確認した。原票は `native_{windows,wsl}_probe.json`、
再現スクリプトは `native_probe.py`。Windows原票にwheel SHA256を記録した。
既存配布物・既存配布wheelは上書きしていない。

user OK後は、このwheelを `D:/puyo_analyzer/packaging/native_build/wheels/` の同名wheelと差し替え、
[同梱手順](B3_REALTIME_2026-10-06.md) の `--b3-model` と `--b3-timing-table` 付きで組み立てる。
今回の接続試験には上記v3モデルを指定可能。モデルの世代名を変更して上書きしない。
[OBS手順](OBS_TEST_B3_2026-10-06.md) に従い、`b3_config.json.search.version` とモデルSHAを確認する。
OBS起動・描画確認と配布物の最終組立は今回実施していない。

## 前回p99の原票分解

対象は旧探索の `b3_realtime/live_30min/{b3_timing.json,b3_updates.jsonl,metrics.json}`。
新探索の性能へ流用しない。`OUT/audit_old.py` は既存原票を読むだけで、
出力 `old_timing_breakdown.json` に元ファイルSHA256・更新契機別統計・遅い上位20行を記録した。

- 全4,203更新からwarmup114更新を除外し、frame 18,000以上126,000未満の4,089更新を再集計。
  特徴p99=1,960.367413ms、最大5,989.686194msと原集計を再現。1秒超は102件。
- p99以上は41件、特徴合計111,844.095ms（全特徴423,304.784msの26.42%）。
  全41件が発火系bitを含み、式40、盤面変化9、heartbeat14、設置1、境界0。
  bitは重複する。発火系全329件のうち41件であり、「境界直後だから冷キャッシュ」とは特定できない。
- 同41件の単行推論は合計140.180ms、p99=6.569ms。評価区間全体は125,262.086ms、
  特徴・推論を引いた残りは13,277.811ms。推論が長い尾の主成分ではない。
- `evaluation_cpu_sec` は評価所有**スレッド**の `thread_time()`。
  同41件のCPU合計112,563.817ms、壁時計との差12,698.270ms。
  評価区間の89.86%はスレッドCPU時間で、待ちだけで長い尾を説明できない。
  これは評価全体の値であり、探索だけのCPU時間ではない。
- 最遅frame 73,450（1224.1667秒、式+発火+盤面変化）は、特徴5,989.686ms、
  推論4.089ms、評価全体6,512.305ms、評価CPU5,989.166ms、残り518.530ms。
- 原票に探索/静止指標/台帳別の時計、cache hit/miss、GCイベント、各更新のOS待ちがない。
  **冷キャッシュ・台帳・GC・CPU競合それぞれの寄与msは復元不能**。
  `processes_mid.txt` にCPU 93.6〜106%のPythonが8本あるため他処理の併走は確認できるが、
  特定更新の遅延への寄与は確定できない。再計測なしで原因名を付けた推定値は出さない。

## 認識への影響と分離案（未実装）

`run_live_pipeline_20260928` → `RecognitionBridge.memory_sample` → `LiveB3.update` →
`RealtimeRows.update` → 状態/台帳更新・`_make_row` が評価側の単一スレッドで同期実行される。
認識は `live_process.recognition_worker` の別processであり、同じスレッドではない。
ただし認識側 `run_recognition_input` の `queue.put` と容量30のFIFOにより、
評価停滞→キュー満杯→認識送信待ち→次の取得が遅れる経路がある。
`VideoFileSource.__iter__` は予定時刻に追いつくためフレームを捨てる。
既存の任意非同期送信も有界バッファ満杯時は背圧へ戻る。
旧原票はcapture drop 44,623/54,000（82.64%）を記録しているが、B3 OFF対照がないため
このうちB3による増加分は確定できない。

分離する場合は既定OFFの任意設定とし、重い探索・特徴・推論を別processへ移す。
認識イベント・NEXT・台帳の状態更新は順序どおり全件適用し、入力通知を最新1件へ間引かない。
計算要求だけを独立した不変スナップショットとして待ち1件の上書きスロットへ送り、
実行中1件+待ち最新1件までに制限する。共有可変GameStateやLRUキャッシュを別threadから触らない。
結果に入力epoch・試合ID・世代・観測時刻を付け、旧試合/期限切れ/順序逆転の結果を捨てる。
通常表示の保持・失敗時0.5秒契約とR由来死亡証明は表示側で独立して更新し、
結果未着が死亡確定・SSE周期を止めないようにする。境界と入力切替で要求/結果を失効させる。
Rust APIはGILを解放するが、Python状態・台帳・静止指標もあるためprocessを第一案とする。
状態更新自体の重さは未分解なので、そこが重い場合は全イベントFIFOを持つB3状態所有processと
最新スナップショットの計算processを分ける。非同期化は今回の特徴一致検査とは別途検証が必要。

## 静かな時のOFF/ON比較手順（今回は実行しない）

他の収集・学習・テスト・ビルドを停止した状態で、同一ホスト、同一動画ハッシュ、同じ認識設定、
CPU/GPU、スレッド数、stride、ウォームアップ、区間、SSE条件を固定する。
300〜2100秒・事前60秒warmupの[前回コマンド](B3_REALTIME_2026-10-06.md)を使い、
まずB3 OFF（`--b3-judgment` とB3引数を付けない、設定JSONもfalse）、完了後にONを逐次実行する。
ONには同じモデル・時間テーブルを明示し、出力を `quiet_off` / `quiet_on` に分ける。
両runのコマンド・設定・資産SHA・開始終了時のCPU負荷と実行中processを保存し、
途中に重い他処理が入った組は比較に採用しない。

両方の `metrics.json` で、同じ54,000予定フレームを母数にcapture drop件数/率と差（ON−OFF）、
認識件数、取得→SSE、評価待ち、キュー最大・末尾残・drainを比較する。
ONでは `b3_realtime_report` による特徴/推論p95・p99、設置→SSEも記録する。
OFFにはB3原票がないため同reportを使わない。ブラウザ/OBSを含める場合は両方で同じ条件にする。
原因分解には事前に状態/台帳・探索・静止指標・行組立ごとのwall/thread CPU、
cache hit/miss、GC開始終了、認識queue.put待ちを追加し、両runで同じ計装条件を使う。
今回の原票にない項目を後付けで測定済みと扱わない。

## 検証結果

B3関連と前回同範囲の接続テストは448件成功、失敗・skipなし（`OUT/tests.xml` / `tests.log`）。
内訳の対象は `tests/test_b3*.py`、`test_live_b3_measurements.py`、
`phase_j/test_live_bridge.py`、`test_live_b7.py`、`test_packaging_closure.py`、
`test_run_live_pipeline_20260928.py`。既存DataFrame構築のPerformanceWarningが12件ある。
Rustは `cargo test --release -j 2 b3::` の8件成功（`rust_tests.log`）。
隠し段おじゃまのB3専用規則と従来シミュレータの挙動維持も含む。
全プロジェクトの範囲外テストや性能ベンチは実施していない。
