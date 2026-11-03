# row1 v2 短区間の検証手順（段3のみ）

既定OFF・未採用。全区間の品質門（段4）は実行しない。PC1での128保存画素の結果は
`docs/ROW1_V2_saved_comparison.json`。これは10Hz冷起動の診断で、以下の合格の代用ではない。
この手順の共有コピーは `share/live_redesign/ROW1_GUARD_V2_PC2.md`。
inboxには置いていないため、共有への配置だけではジョブを起動しない。

## 固定コードの受取り

bundle: `share/live_redesign/live_row1guard_v2.bundle`。
前提コミット: `c2cecacaae9537f2ad6a1180f25541e77ac76b45`。
実HEAD・SHA-256は隣の `live_row1guard_v2.receipt.json` を照合する。
PC2で、既存eg作業ツリーを変更せず、新しい作業ツリーを作る例:

```bash
repo=/mnt/d/puyo_analyzer/eg_row1_offoff_nondeterminism
bundle=/mnt/pc2_attached_d/puyo_analyzer/share/live_redesign/live_row1guard_v2.bundle
code=/mnt/d/puyo_analyzer/row1_v2_short_code
git -C "$repo" bundle verify "$bundle"
git -C "$repo" fetch "$bundle" HEAD:refs/heads/row1-v2-short-review
git -C "$repo" worktree add --detach "$code" refs/heads/row1-v2-short-review
git -C "$code" rev-parse HEAD
```

モデル・較正資産はegで復旧済みの物を再利用し、SHAを記録する。
新しい作業ツリー内に無い資産を理由にHSV等へ黙ってフォールバックしない。
`done/eg_row1_offoff_nondeterminism/restored_assets.json` と実効設定を照合する。
認識・評価workerで `src.recognition_pipeline.__file__` が新コードを指すことも記録する。

## 実行台と設定

PC2ローカルの `/mnt/c/puyo2/codex_work/eg_row1_offoff_nondeterminism/` の
`entry.py`, `controller.py`, `supervise.py` と関連ファイルを新しい作業ディレクトリへ複製する。
旧ファイルは上書きしない。コードrootを上記 `code`、出力rootを
`/mnt/d/puyo_analyzer/tasks/row1_v2_short/` に変更する。
egの色補正・VerifiedVideo/DeviceSession・実live認識/評価/表示・資源監視・種の複製は維持する。
entryのソース指定とcwdの資産参照を区別する。実行入口の形式は従来通り:

```bash
nice -n 19 taskset -c 6,7 /mnt/d/puyo_analyzer/venv/bin/python \
  /mnt/c/puyo2/codex_work/row1_v2_short/entry.py \
  --config /mnt/d/puyo_analyzer/tasks/row1_v2_short/game36/off/config.json \
  --video /mnt/d/puyo_analyzer/tasks/row1_v2_short/video_65.mp4 \
  --output /mnt/d/puyo_analyzer/tasks/row1_v2_short/game36/off/out
```

この入口は監視するcontrollerから起動し、各条件1プロセス・直列、BLAS/OMP/MKL各1、
CPU6–7/nice19（WindowsならBelowNormal）。Windows/WSLの空き各6GiBを確保し、
下回ったら自分のプロセス群だけ停止する。既存ジョブ、WSL設定・マウント・常駐・電源には触らない。
長時間起動は既存superviseかsetsidによるdetachを使う。pushはしない。

- `done/eg_row1_offoff_nondeterminism/{early,game36}/a/config.json` をそれぞれ複製する。
  OFF/ONの差は `live_row1_guard` だけ。認識workerでは `enable_row1_col2_guard` だけ。
- `recognition_audit=true`、realtime=false、全30Hz。R1照合器はOFFのまま。
  認識設定はbuild t OBS実機保存版の代用であることを受領票に明記する。
- 較正種は `done/eb2_row1_guard_quality_gate_retry/seed_original.json` を各条件に新規複製。
  SHA-256 `effb5dca5d7766388be96fa6ccdf8360215007ddce2270abd1206023bbfc09d1`。
  OFF実行後のprofileをONの初期値にしない。
- 動画65はID `g_bFvlXeJks`、既存メタデータはeg成果の `video_65.info.json`。
  原動画SHA-256 `46bf23234d0816d05d2dbd123139ad7245ad3bedf03ae8d40fdd13938a8f0316`。
  削除済みなら既存のyt-dlp手順（format 136）で取得し、同SHAを確認する。
  使用後は今回取得分を削除し、URL・メタデータ・SHA・削除票を残す。

## 区間と既知失敗の被覆

1. **early:** start_sec=0, end_sec=200, warmup_sec=1。
   動画先頭から。元コマf5184〜5212の盤面有無差を含む。
2. **game36:** start_sec=2200, end_sec=2375, warmup_sec=1。
   実読開始2199秒、書出し2200秒。試合開始より約99秒前からのegと同じ履歴。
   f70734（2357.8秒）の勝率と、#8のf69864〜69930（2328.8〜2331.0秒、両端込み67コマ）を採点。
   #8を同じ走行内で採点できれば重複走行しない。
3. #8がOFFで67/67赤を再現しなければ合格扱いしない。
   eb2の正規短区間 start_sec=2290, end_sec=2350, warmup_sec=1 に限定して再現確認する。
   無再現のままONが空という理由で幻除去成功としない。
4. **着地8コマ基準:** 上記区間内の実着地を画像で特定して測る。
   特に旧版で9コマだったvideo94/1P/f49671（黄、±1コマ）は未解決の回帰標本。
   この標本も測る場合、eb2保存displayの試合境界から当該試合開始前に切り、終端は1670秒程度。
   全動画を流さない。旧OFFのf49674初公開を再現できることを先に確かめる。
   動画94のSHAは `7287cab5b4942eefdc44a1502ac60fadbb3b05bad18d361cd0226321562b8e3a`。
   この標本を未実行なら「8コマ上限を全件クリア」と書かない。

まず新コードOFFを保存eg/a（early/game36）と照合する。フレーム集合・公開盤・CNN・状態・
active・ready・score・表示が同じか記録する。構成差や欠落があればONの評価と区別する。
元のOFFはf70734で31.691207%、旧ONは87.658979%。比較はこの元コマで行う。

## 作動ログと下流の追跡

`recognition.row1_guard.jsonl` のversion=2を確認する。OFFはshadow=true、ONはfalse。
旧フィールドに加え `evidence.signals`（formula/next_moved/ojama_falling）、
raw_top、after_top、held_since、suppressed、restoredを記録する。
解除理由は `chain_started`（掛け算式実読）、`next_moved`、`ojama_falling`、
`raw_disappeared`、`top_empty`、`no_board`。時間経過による解除はない。

残る対象外差は、最初の作動から順に追う。同側23コマ近接だけで因果を断定しない。
必要な観測は実entryの認識worker内で毎コマ読み取り専用に保存する:

- `_apply_row1_col2_guard` 前後の公開盤、入力raw_cnn_board、側別の3合図。
- `update` 後の `_sm_1p/_sm_2p.context.confirmed_board`、`_prev_confirmed_1p/2p`、
  `_prev_stable_confirmed_1p/2p`、SideResultのinferred_board・landing_diag・state。
- 公開認識票とdisplay_traceのframe/source/raw_probability/probability/game。

公開盤は `recognition_pipeline.py` のcycle71dで `_prev_confirmed` に入り、次コマの
連鎖検出・式による早期発火へ戻る。`ctx.confirmed_board`が同じでも全内部状態同等とは限らない。
本変更はこの配線を切っていない。較正へのフィードバック、置き推定、B3着地検出、
表示平滑も追う。監査NPZのrawはcnn_boardでありガード入力raw_cnn_boardの代用にはしない。

## 集計・判定・納品

- 全対応コマの公開盤セル差を、盤面有無／row0／row1 col2／その他へ分け、母数も出す。
  v2の直接書換えはrow1 col2のみ。row0の差も対象外として数える。
  対象外差0を目標とし、残るなら全座標と作動ログ・内部履歴による理由を出す。
  理由不明の差を除外して合格扱いしない。
- #8はOFF赤67/67の再現と、新ON非空/67・赤/67を記録する。OFF未再現は被覆外。
- 着地は接触参照frameから初公開までの**元動画コマ数**をOFF/ON別に出す。
  ONが8コマを超える、または公開欠測なら合格としない。合図から解除までの0コマと混同しない。
- 表示差の最大/P95/符号反転/差コマ数、f70734の両値、最大差コマ前後の作動を記録する。
  公開盤一致後の差も捨てない。許容値を新設して勝手に合格にしない。
- 条件ごとの実効設定・資産/種SHA・コードHEAD・読取開始/採点開始・欠落・資源票を保存。
- 新しい成果ディレクトリへSUMMARY.md（40行以内）、DETAILS.md、comparison.json、
  対象外差一覧、着地参照画像と採点、作動JSONL、recognition/display原票を納品する。
  未測定・被覆外と合格を分ける。段3の結論を出して終了し、段4へは進まない。
