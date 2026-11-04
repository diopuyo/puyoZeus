image_free_recog2 の並列比較運用（2026-10-07）

`scripts/image_free_parallel.py` は固定ソースを外側から呼び出すWSL専用監視器。
比較側のidentityはsrc/scripts全体をハッシュするため、既存チェックポイントを
再利用するジョブのsource内へ新しい監視器をコピーしてはいけない。
ジョブ直下のparallel.pyへコピーし、`--source source --config config.json` で呼ぶ。
全作業が同じoutputのui.joblib/digit_student.joblibを参照し、動画ID別フォルダに保存する。
元config・固定ソース・比較モデル・identity検証を変更しない。

監視器だけが5秒間隔でPuyoLive/PuyoZeusとWindows空きメモリを問い合わせる。
PowerShellの非ゼロ終了・タイムアウト・解析失敗は最初の2回まで直前の停止状態を維持し、
3回連続で全作業グループをSIGSTOPする。初回の監視成功前は起動しない。
成功した監視で条件が解消すればSIGCONTする。各作業の旧ObsPriorityGuardは使用しない。
監視器終了時には子を終了し、監視器強制終了時もLinuxの親終了シグナルで子を終了する。

各作業は数値ライブラリのスレッド数1とCPU affinityで1論理CPUの実行量に制限する。
CPUはSMT兄弟を避けて物理コアを優先する。既定最大4作業だが1監視周期に1作業だけ起動する。
新規起動にはWindows/WSL両方で空き8GiB以上（維持6GiB＋起動予約2GiB）が必要。
実行中も両側を監視し、6GiB未満なら全停止して1作業ずつ終了しメモリを返す。
再実行は確定済みチェックポイントから。最大299フレームの未確定分は再計算する。
動画を途中で区間分割することはなく、STABLE状態を含む連続比較の意味を維持する。

起動例（PC1）:

```powershell
wsl -d Ubuntu -- bash -lc 'setsid -f bash /mnt/d/example_project/b3_data/codex_tasks/image_free_recog2/run_parallel.sh >> /mnt/d/example_project/b3_data/codex_tasks/image_free_recog2/parallel.log 2>&1 < /dev/null'
```

進行はsupervisor.jsonと各動画のprogress.json、確定点はcheckpoint.json。
supervisor.jsonのupdated_atが更新していることも確認する（終了後に残る状態は過去の記録）。
再起動はsupervisor_pidへTERMを送り、プロセス終了とロック解放を確認してから起動する。
workerのexit_codesが非ゼロの場合はその動画のparallel.logを調べる。
全終了後のresult.jsonは合格判定をしない。候補の画像読込禁止など固定比較側の検査を保つ。

PC2が利用可能になるまではPC1の4本を継続する。
PC2作業一式と詳細手順は `D:/example_project/b3_data/codex_tasks/image_free_parallel_pc2/`。
転送先は `\\192.0.2.2\example_share\codex_tasks\image_free_parallel_pc2`。
PC2用configはvideo_d01/video_d02の2本。絶対パス入りpickleの移植を避け0から実行する。
PC2で両動画の進行確認後、PC1ジョブ直下のparallel_control.jsonへ
`{"disabled_videos":["video_d01","video_d02"]}` を保存する。
当該作業は止まり、確定点が残る。空配列へ戻すとPC1で再開できる。
PC2の結果は別フォルダに回収し、PC1のc68/video_d04と動画単位で選択する。

回帰検証:

```bash
python -m pytest tests/test_image_free_parallel.py tests/test_image_free_all.py -q
```
