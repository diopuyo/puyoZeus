# 配布版等価性：固定復号フレームでの切り分け

## 過去の比較条件

`D:/puyo_analyzer/packaging/equiv/cmp_A_T2.json`、`cmp_T2_C1.json`、
`cmp_T2_O1.json`、`cmp_T2_O2.json` は差 0。
torch 対 ONNX の合格は、同じ Windows 環境・同じ動画復号経路によるものだった。
`cmp_T2_wslA.json` は元から差があり、OS 間の復号画素をそろえた合格記録ではない。

同ディレクトリの `decode_diag/RESULT.md` に、Windows 同梱 FFmpeg 4.4 系の
BT.601 変換と Linux FFmpeg 8 系の BT.709 変換の違いが記録されている。
`decode_diag/dump.py` / `recog_ctl.py` は無損失保存した画素を再投入する対照だった。
今回もこの原則を使い、保存形式を連続 BGR バイト列とした。

前回 10/3 の比較は Windows の配布外 FFmpeg DLL と Linux の FFmpeg でそれぞれ MP4 を復号していた。
双方とも backend は FFMPEG であり、FFmpeg 除去後の別バックエンドへの暗黙切替ではなかった。
既に 225 秒から 60 フレーム中 60 フレームで入力画素が不一致だったため、
その出力差だけでは配布版の認識・評価の欠陥と判定できなかった。

## 比較台の修正

- `packaging/equiv_frames.py`: WSL で一度だけ復号し、BGR 列とフレームごとの SHA-256 を保存。
  再投入時は動画デコーダーを使わず、サイズ・絶対フレーム番号・各フレームのハッシュを検査する。
  不一致や未完成のキャッシュは例外にし、動画読込へのフォールバックは行わない。
- `packaging/run_fixed_equiv.py`: 検査 process と spawn 子だけで入力を差し替える。
  配布の OCR ゼロ判定・高速テロップ・高速終了検出を両側で使い、
  前回の等価試験と同じ `--no-split-evaluation --no-async-counter` で全フレーム処理する。
  本番フラグ・資産 MANIFEST・seed・ライブラリ版と認識分解記録を残す。
- `packaging/compare_outputs.py`: 切詰め後に消えていた行数差、1,000 行で打ち切られていた不一致行集計、
  列欠落・CSV ヘッダー差の見逃しを修正。許容差は追加していない。
- `packaging/summarize_fixed_equiv.py`: 入力一致・母数・最初の表示差・最初の認識差をまとめる。
  比較台の単体試験は `tests/test_packaging_equiv.py` の 8 件が成功。

WSL の共有ドライブ読込は 64 MiB / 34.62 秒だったため、同じ BGR 列を標準入力経由で
WSL の `/tmp/puyo_equiv_fixed_20261003/` へ複製して使用した。
これは入力データや評価設定の変更ではなく、保存場所の変更である。
読込時の全画素ハッシュ照合を複製後も行う。

## 条件と結果の証跡

- 入力: `q_7gc4TgFig_first_0_900_20260925_v1.mp4`。
- 固定画素: 225–340 秒、3,450 フレーム。出力: 226–340 秒、3,420 フレーム。
- 同一の配布版 `app/` を Windows 埋込 Python と WSL 開発 venv で使用。
- 本番フラグ・モデル・テンプレート・乱数 seed は変更しない。
- ONNX の門は `docs/PHASE_J_ONNX_PREREGISTRATION_2026-09-30.md` の全パイプライン差 0 を使用。
  微差を許す新しい閾値は設けていない。
- 証跡ルート: `D:/puyo_analyzer/packaging/equiv/`。
  Windows torch: `fixed_win_torch/`、Windows ONNX: `fixed_win_onnx/`。
  `fixed_onnx_result.json` と `.full.json` に全 5 ファイル差 0 を記録。

## 確定した原因と結果

大きな不一致の原因は比較台の復号画素差だった。入力を固定すると、Windows／WSL の
盤面（raw／confirmed）、状態、STABLE、active、ready、得点は全行一致し、
STABLE 出力も双方 2,249 行になった。設定、資産 MANIFEST、seed、本番フラグも一致。
配布版の欠陥は確認されず、本体・本番設定・モデルの修正や再組み立ては行っていない。

残る OS 間の差は torch CPU 演算の数値差であり、ONNX 由来ではない。
最初の表示差は行 0（226.0 秒）の `display_adv`:
Windows `-3.5042638091317158`、WSL `-3.5042628192997762`。
分解記録の最初の static（inputs の物理行 22）は特徴量が同一で、M0 確率だけが
Windows `0.5897228161827781`、WSL `0.5897228495235004` と異なった。

`packaging/probe_equiv_m0.py` で記録から同一盤面・NEXT を再構成し、static_key を照合した。
実行時の CPU 1 スレッドで直接 M0 を呼ぶと、双方の記録確率をそれぞれ完全再現した
（`m0_win1.json` / `m0_wsl1.json` の `reproduces_record=true`）。
重み SHA-256 は双方 `f2192f695ad40d34f6919dcb31f3cfed0875925ded0c48069f0ab33f09b7b85f`。
Windows は torch `2.5.1+cpu`、WSL は `2.5.1+cu121` の CPU 実行。
forward hook の入力と最初の Conv は完全一致し、最初に差が出る層は
`side_encoder.color_features.1`（GroupNorm）、最大絶対差 `4.76837158203125e-7`。
したがってフラグ・テンプレート・モデル重み・乱数・設定読込ではなく、
同一入力に対する torch CPU 実装の数値差と特定した。

| 比較 | 結果 |
|---|---|
| 実際に投入した画素 | 全 3,450 フレームの SHA-256 一致 |
| 完走／drop | 各 3,420 フレーム／0 |
| Windows torch 対 ONNX | 全 5 ファイル差 0、従来のパイプライン門に合格 |
| Windows 対 WSL の認識分解記録 | 対象全列・全行一致 |
| Windows 対 WSL の表示 | 1,213 / 3,420 行に数値差、最大絶対差 `3.34698432169489e-5` |
| Windows 対 WSL の厳密比較 | 不一致集計 10,304、厳密一致は未達（許容差は新設しない） |
| 比較台の単体試験 | 8 passed |

不一致集計はファイルごとの行差・セル差の合計で、独立フレーム数ではない。
OS 比較の全記録は `fixed_os_result.json` / `.full.json`、
完走データは `fixed_win_torch/`、`fixed_win_onnx/`、`fixed_wsl_local/`。
`fixed_wsl_torch/` は共有ドライブ I/O のため中止した途中出力で、判定には使っていない。
要約 JSON と M0 再現記録は `docs/equiv_20261003/` にも保存した。

## 再実行

各環境の Python で次の順に実行する。`APP` は配布物の `app`、`CACHE` は
`fixed_20261003_raw`（WSL ではその同一コピー）、`OUT` は新しい出力先、
`VIDEO` は上記入力名のパス。復号は最初の一度だけで、比較実行では動画を開かない。

```text
python packaging/run_fixed_equiv.py APP CACHE OUT VIDEO --backend torch --port 8871
python packaging/run_fixed_equiv.py APP CACHE OUT_ONNX VIDEO --backend onnx --port 8872
python packaging/summarize_fixed_equiv.py OUT OUT_ONNX RESULT.json
python packaging/probe_equiv_m0.py APP OUT/inputs.jsonl.gz M0.json --threads 1
```
