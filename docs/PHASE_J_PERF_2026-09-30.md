# Phase J 配布版 (Windows) の認識速度 — 実測と改善 (2026-09-30、コーダ)

branch `claude/realtime-packaging-20260930`。指示: rt の `fast_terminal` を配布既定へ、Windows 認識 P95 ≤ 33 ms/frame (30 Hz) を目標に、決定を変えずに速くする。
**割り算の単位**: 認識時間は「認識が実際に処理した frame」(先頭 60 frame は暖機として除外、n=900 が既定)。PNG 読込は計時外。
**負荷の注意 (全体)**: 下記の「同時負荷」欄に、各測定中に動いていた別ジョブを記す。**最初の 2 本 (base1/base2) と WSL 1 本を除き、別 agent の WSL ジョブ (python 2〜3 本、nice 19、Windows 全体 CPU 20〜45%) が並走していた**。
絶対値は「その負荷下」の値で、同一 run 内の対比 (交互 A/B) を主根拠にする。

## 1. 取り込みと配布既定 (依頼 1)

- rt `claude/realtime-phase-j-20260928` (c8b8996) を merge (競合なし、1bc6441)。
- 配布既定 (`src/phase_j/launcher.py PIPELINE_FIXED`): `fast_terminal=True` (rt 実測: 491,928 frame で決定差 0)、`async_notice_queue` は **OFF のまま** (キーを出さない)。
- 再ビルド: `python packaging/build_bundle.py --reuse-python --with-onnx` (約 35〜60 秒)。MANIFEST 照合・launcher 試験は通過 (配布用 python 81 passed / 1 skipped、WSL 521 passed / 2 skipped + 221 passed)。
- **未実施 (指示で一時停止)**: `--zip` の最終再ビルド。`D:\puyo_analyzer\packaging\build\PuyoLive\` (展開版) は最終コードだが、`PuyoLive.zip` は本作業の冒頭 (ONNX なし・改善前) のもの。発火前予測のジョブ終了後に
  `python packaging/build_bundle.py --reuse-python --with-onnx --zip` を実行すること。全 pytest (4,400+) は未実行 (触ったモジュールの試験のみ)。

## 2. Windows 認識速度 — 閑散時の再測定 (依頼 2)

同一 PNG 960 枚 (zenchi 5880.566 から stride2 = 30 Hz、WSL の BT.709 復号で書き出し、測定後に削除)。1 スレッド設定 (`cpu_threads=1`)、CPU のみ。

| 環境 | 同時負荷 | P50 | P95 | P99 |
|---|---|---|---|---|
| Windows 配布 python (改善前、fast_terminal ON、torch) base1 | WSL python 1 本 (軽い) | **33.4** | **41.6** | 45.6 |
| 同 base2 | 同 | 33.8 | 44.2 | 51.2 |
| WSL 同一 PNG (torch) wsl2 | WSL python 1 本 | 25.9 | 33.1 | 38.7 |

- 「Windows で 60〜68 ms」は**負荷下の値**だった。閑散時の Windows は WSL の約 1.29 倍 (P50)。
- 決定 digest (result の盤面・状態・得点など全フィールドを正準化) は **Windows と WSL で 960/960 frame 一致** (測定器の健全性確認)。同一 config の Windows 2 run 間も 0 差。digest は frame 毎に全て異なる (960 種、定数出力ではない)。

## 3. 時間の行き先 (プロファイル)

cProfile (200 frame、プロファイル自体で約 +10 ms 膨らむ。Windows 45.0 ms 対 WSL 35.1 ms) と、同一 frame 列での Windows−Linux 差の分解:

- 差の約 4.6 ms は**大きい配列を作る・読む関数** (`zeros_like`・`integral2`・`idft`・`mulSpectrums`・`dft`・`matchTemplate`・`cvtColor`・`astype`) に集中する。要因の有力候補は大きい配列の確保 (`packaging/perf/microbench.py`、1 スレッド、Windows 対 Linux の 1 回あたり): `zeros_like` 1080p 11.1 倍、`frame.copy()` 4.9 倍、`integral2` 6.8 倍。
  Linux は glibc が解放済みの大きいブロックを再利用するが、Windows は毎回 OS から零ページを取り直す (推定。ページフォールト計測はしていない)。テロップ検出 (FFT 相関と正規化の一時配列約 25 個) が最大の被害者。
- 残りは全体に一様な約 1.2 倍 (Python ループ 1.18 倍、小さい numpy 1.23 倍、cv2 の 1080p 変換 1.6〜1.9 倍)。CPU 周波数・環境差と考えられる (未切り分け)。
- `puyo_core` (Rust) は `--release` (`opt-level=3`、`lto`) でビルドされる手順で、1 呼出 8.3 µs 対 Linux 6.0 µs (1.38 倍) と全体の一様な差と同程度 (デバッグビルドの兆候なし。バイナリの最適化フラグ自体は未検査、ビルド手順書の記述に基づく)。
- torch CPU の畳み込みは本測定では Linux (cu121 版) より Windows のほうが速い (0.66 倍)。CNN は ONNX へ切替可能 (下記)。
- **E コア問題 (最大の発見)**: このマシン (i7-13620H、P 6 コア + E 4 コア) で認識 process を E コア 1 つへ固定すると P50 **75.6 ms**、P コアなら 33.7〜36.3 ms (各 n=400、2 回ずつ)。
  外部負荷 (busy process 4 本) を与えると、何もしない既定では P50 **56.7〜65** ms (P95 73〜82) に悪化 (実質 E コアや HT 兄弟へ寄る)。これが従来の「60〜68 ms」の正体。

## 4. 改善 (依頼 3) — すべて決定不変を digest で確認

| 変更 | 場所 | 内容 | 効果 (Windows) | 決定 |
|---|---|---|---|---|
| **performance_cores** (新フラグ、配布既定 ON) | `src/phase_j/live_win_cores.py`、`live_cpu.apply_runtime` | 認識 process だけを最大効率クラスの CPU set (`SetProcessDefaultCpuSets`、ソフト) へ。非ハイブリッド CPU・Windows 以外は何もしない (適用 CPU set 数を stderr へ必ず出力) | busy 4 本の下で P50 56.7〜58.5 → **44.7〜45.7**、P95 73.2〜76.7 → **55.3〜57.9** (3/3 組) | 計算に触れない |
| **fast_telop** (新フラグ、配布既定 ON) | `src/telop_detector.py` | 1/4 縮小 ROI の粗 NCC が床 (0.30) 未満のテンプレートは全解像度照合を省く (`fast_terminal` と同方式) | 閑散〜軽負荷で P50 **-6.5〜-7.4 ms**、P95 -6.1〜-7.2 ms (3/3 組、n=400) | 下記 §5 |
| **ONNX CNN** (配布既定 ON、同梱ビルド時) | `launcher.apply_cnn_backend` | `models/onnx/index.json` が同梱されていれば `PUYO_CNN_BACKEND=onnx` を設定 (利用者の明示を優先)。要求して使えなければ `cnn_onnx` が例外 (黙って torch へ戻さない) | P50 -0.7〜-4.0 ms (3/3 組、平均 -2.6) | 事前登録 (2,073,168 セル argmax 不一致 0) + 本 digest 0/960 |
| テロップ `scores` の in-place 化 | `src/prepared_template.py` | 元式と同じ演算順のまま一時配列を減らす。基準実装 `scores_reference` を残し試験で bit 一致を固定 | 2 テンプレート 1 frame で約 -1.4〜-1.9 ms (120 frame の微小測定) | 実 frame 120 x 2 テンプレートで `array_equal` (不一致 0) + 試験 21 件 |
| `_board_hsv_frame` の `np.zeros` 化 | `src/image_reader.py` | `zeros_like` を `np.zeros` へ (値は同一、Windows で遅延零ページ) | 微小測定 -0.65 ms | 値同一 |

**入れなかったもの**: cv2 スレッド 2、torch スレッド 2 (雑音下の 1 回だが悪化: P50 43.4)、GC 調整 (差が雑音内)、`SetPriorityClass` ABOVE_NORMAL (busy 負荷下で効果なし)。
`_board_hsv_frame` の HSV バッファ再利用は、`_live_hsv_pixels` が view を保持するため危険と判断して見送り。

### 最終構成の交互 A/B (改善前 = 1bc6441 の src + torch + `fast_terminal` のみ、改善後 = 全部)

条件: 各 n=900、3 組を交互に、同時負荷は WSL の別 agent ジョブ (python 2〜3 本)。

| 組 | 改善前 P50 / P95 / P99 | 改善後 P50 / P95 / P99 |
|---|---|---|
| 1 | 37.0 / 52.0 / 66.1 | **25.2 / 31.9 / 37.1** |
| 2 | 38.3 / 51.9 / 66.9 | **25.8 / 33.0 / 37.7** |
| 3 | 41.4 / 58.3 / 68.3 | **25.1 / 32.2 / 38.6** |

- **決定 digest: 改善後 対 改善前 = 960 frame 中 0 差 x 3 組、改善後 対 WSL torch = 0 差 x 3 組。**
- 目標 P95 ≤ 33 ms: 31.9 / 33.0 / 32.2 で**ほぼ達成 (余裕は薄い)**。P99 は 37〜39 ms で 1 frame 枠 (33.3 ms) を超える (原因の大半は死亡検出の全域照合へ落ちる frame と、盤面読取が重い frame。段別計時: 遅い 5% は死亡検出 +6.2 ms、盤面読取 +3.2 ms (P50 以下の frame との差))。
- 段別 (改善後、n=900、計時付き): 盤面読取 12.7 ms、死亡検出 2.7、得点ゼロ検出 1.6、NEXT 検出 1.3、NEXT スライド 1.0、テロップ 0.65、その他 5.7 (P50 25.2 / P95 32.5)。
- 外部負荷を人工的に与えた場合 (busy 4 本): 改善前 P50 63.6〜71.2 / P95 82.5〜89.2 → 改善後 **40.9〜42.2 / 51.7〜53.9**。busy 8 本: 73.0 / 92.5 → 48.5 / 63.2。
  **重い競合下では 33 ms に届かない**。その場合の必要な間引きは (推定) 認識 15 Hz (2 frame に 1 回、枠 66.7 ms)。レートは変更していない。

## 5. fast_telop の決定同一性の証拠と限界 (重要)

- 全数照合 (`scripts/scan_telop_prefilter_b21.py` → `packaging/perf/analyze_telop_scan.py`): zenchi 全長 211,008 + video_38 99,000 + video_100 94,860 + video_121 87,060 = **491,928 frame**
  (stride2、1080p。B20 と同一母集団) x 2 テンプレート = **983,856 テンプレート・frame**。
  全解像度の本番 `peak` 値と 1/4 粗スコアを保存し、床 0.10〜0.50 の各値で `TelopDetector.detect` の (is_visible, bbox) を再構成して比較: **全床で決定差 0**。床 0.30 で全解像度経路へ進むのは 0.023% (223 テンプレート・frame)、99.977% を省略。
  粗スコアの最大 (陰性): 0.273 (テンプレート 0) / 0.352 (テンプレート 1)。
- **限界: 491,928 frame に可視テロップ (全解像度スコア ≥ 0.55) は 0 件**。テンプレートは「チャレンジャー リーグ 30先」放送 (video_01 の 1670 秒付近、動画は削除済み) 由来で、この母集団に陽性が無い。
  したがって「陽性を誤って不可視にしない」ことの実データ証拠は**無い**。代わりに合成陽性 (`packaging/perf/telop_synthetic_margin.py`): 実背景 frame へ実テンプレートを
  フェード (α 0.4〜1.0)・ぼかし・±6% 拡縮・明るさ ±40・ノイズ σ≤8・位置ランダムで貼った 400 試行 = 800 対、うち全解像度 ≥ 0.55 の陽性 **216 対**で、粗スコアの最小 **0.517** (床 0.30 に対し余白 0.217)、
  全解像度−粗スコアの最大乖離 0.163、**床での誤スキップ 0 件**。合成は実放送の劣化を網羅しない。
- 影響範囲: 高速経路は `TelopResult.template_name / score` を不可視 frame で省く (実消費者は `is_visible` と `bbox` のみで、grep で他の参照無しを確認)。配布版の利用者の映像にはこのテロップは通常出ない (放送 overlay 由来)。
  もし陽性の取りこぼしがあれば、テロップ下のセルが UNKNOWN 扱いにならず誤読が増える方向。**user 判断で外すなら `PIPELINE_FIXED` から `fast_telop=True` を消すだけ** (既定 OFF、bit-identical)。

## 6. OBS 仮想カメラ入力 (依頼 4)

手順: OBS Studio に専用プロファイル (1920x1080@30、出力 1920x1080) とシーン (メディアソース、zenchi 5830〜6210 秒を無音・コピーで切り出しループ) を一時作成 → `--startvirtualcam` → 配布物 (ONNX 同梱の最終ビルド) を
launcher 経由 (`packaging/perf/run_obs_measure.py`、`dshow_color_correction=601to709`、index 4) で 330 秒。認識監査と SSE 受信記録 (`recognition_audit`/`runtime_audit`) を計測専用に有効化。
**OBS の設定は復元済み** (`global.ini`/`user.ini` は退避コピーへ戻し、`basic/` は退避と `diff -rq` で同一、一時プロファイルとシーンは削除。「無題」は未変更)。切り出した動画と PNG も削除。

**最重要の発見 (性能ではなく入力の門番)**: 既定のまま走らせると、**認識に渡った frame は 330 秒中 1,815 (予定 9,900 の 18%)**。`PuyoScreenVerifier` (1 秒周期の「ぷよ画面確認」: 盤面枠 Hough + 両側得点 OCR) が
実際の対戦 frame を頻繁に不合格にし、`no_puyo_screen` → 再較正 (「色を較正中」) を 40 回繰り返した (1 回不合格になると次の確認まで 1 秒 frame が渡らない)。
オフラインで同じ検証器を掛けた合格率: OBS 仮想カメラ実 frame 216 枚中 **62 (601→709 補正後、補正なしは 38)**、同区間の BT.709 PNG 240 枚中 91 (38%)、他動画の等間隔標本 60 枚中 19〜29。
不合格の内訳 (補正後 154 件のうち集計した上位 132 件): 盤面枠の格子が 1 側以上で未検出 118 件、得点 OCR が 1 側以上で読めない 20 件 (連鎖式表示中など。重複あり)。**配布版の実運用で表示が頻繁に止まる恐れ**。これは認識速度と無関係の別問題で、`live_device.py` の設計 (別 agent/user 判断) に関わるため触っていない。

性能は、検証器を診断用に常時合格へ差し替えて (`PERF_BYPASS_VERIFIER=1`、診断であり品質合格ではない、`diagnostic_verifier_bypassed`) 再測定した:

| 項目 (330 秒、ready 後 約 312 秒) | 値 |
|---|---|
| 認識が処理した frame | 6,670 / 予定 9,354 (= (330-18.2) x 30) = **71.3%**。処理されなかった 28.7% は DirectShow 側の取りこぼし (source の `dropped`=0 で計上されない。`event_priority` の BufferedCapture を使えば計上される) |
| frame 通過率 (30 秒窓) | 13〜27 fps (窓ごとに 453, 658, 787, 587, 819, 819, 806, 409, 371, 452, 496) |
| 認識 P50 / P95 / P99 (ready 中 6,670 frame) | **30.4 / 65.6 / 83.9 ms** |
| 取得待ち P50 / P95 | 5.6 / 9.2 ms |
| 捕捉 → 認識完了 P50 / P95 | 36.1 / 74.6 ms |
| 捕捉 → SSE (サーバ送出、n=427 DTO、最大 2 Hz) P50 / P95 / P99 | 68.9 / 184.4 / 491.2 ms |
| 捕捉 → SSE (購読側受信、n=477) P50 / P95 / P99 | 67.3 / 185.5 / 461.1 ms |
| CPU (16 論理、n=124 サンプル) | 全体 平均 44.1%・P95 77.8%。認識 process (最大の python) 平均 86% / P95 106% (論理 1 個分)、OBS 平均 39%、評価 worker 等 10〜26% |

- **この 5 分の測定は負荷が汚れている**: 別 agent の WSL ジョブ (測定開始時は pytest 1 本、終了時点で発火前予測の 16 プロセス) が並走し、全体 CPU は平均 44%。§4 の PNG 測定 (P50 25 / P95 32) より遅く、認識 P50 が 30.4 / P95 が 65.6 になった。
  加えて live の認識 thread は色行列補正 (`cv2.transform` 1080p、Windows 1 スレッドで P50 3.4 / P95 6.7 ms) と capture の読出・pacing を同じ thread で行う (PNG 測定にはない)。
- 実時間 30 Hz は、この負荷下では**維持できていない** (処理 71%)。閑散時の再測定は、発火前予測ジョブの終了後に `run_obs_measure.py` で行うこと (再実行のみ、コード変更不要)。
- 使わなかった run (記録のみ、`packaging/perf/obs_run_clip1_nongameplay`): fcXG の切り出しは配信画面の切替が多く、入力が「ぷよ画面なし」を繰り返した。

## 7. 再現・成果物

- 測定器: `packaging/perf/bench_recognition.py` (PNG 連番、digest、段別計時、cProfile)、`microbench.py`、`run_obs_measure.py`、`telop_synthetic_margin.py`、`analyze_telop_scan.py`、`scripts/scan_telop_prefilter_b21.py`。
- 生データ: `D:/puyo_analyzer/packaging/perf/` (`*.json` 結果、`*.digests.json`、`*.times.json`、`*.stages.json`、`telop_scan/`、`obs_run*/`)。
- 試験: `tests/test_live_win_cores.py` (9)、`tests/test_prepared_template_inplace.py` (21)、`tests/test_telop_fast.py` (4)、`tests/test_launcher.py` (+3)。
- 既知の制約: 測定機は i7-13620H (ハイブリッド CPU)。非ハイブリッド CPU では `performance_cores` は何もしない (効果は E コア問題が無い分だけ小さい)。
