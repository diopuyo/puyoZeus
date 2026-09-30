# Phase J リアルタイム・オーバーレイ 配布パッケージ計画 (2026-09-30)

基点: `claude/realtime-phase-j-20260928` (7992dc9)。作業ブランチ: `claude/realtime-packaging-20260930`。
前提: memory `project_realtime_input_decisions_2026-09-28` (配布前提・機種非依存・OBS 仮想カメラ主入力・
入力判別 3 段・色ウォームアップ・CPU 既定)。

**測定の注意**: 本書の数値はすべて 1 台 (Windows 11 / RTX 4060 Laptop / D: は SSD) での実測。時間は「秒」、
サイズは「MB (10^6 バイト)」。起動時間は各条件 1〜2 サンプルで、他エージェントが同時に動いていた。
統計ではなく桁の目安として読むこと。

## 1. 結論

**採用方式: 埋め込み Python 3.12.3 (python.org の embeddable zip) + CPU 版 wheel を直接展開 + zip 配布。**
venv も PyInstaller も使わない。`PuyoLive.bat` → `python\python.exe -m src.phase_j.launcher`。

| 項目 | 実測 |
|---|---|
| 展開後サイズ | **800.7 MB / 13,246 ファイル** (python 781.2、app 18.6、LICENSES 0.9) |
| zip サイズ | **270.1 MB** |
| 組み立て所要 (python 込み全工程、初回) | 約 17 分の見込み (実測は 2 回に分かれた: 導入・app 組立 753.5 秒 + 事前コンパイル/削減を含む再構成 324 秒。ダウンロード別。1 回通しの実測は未取得) |
| 組み立て所要 (`--reuse-python`、app のみ更新) | 23〜28 秒 |
| zip 展開 (Python `zipfile`、12.6 千ファイル) | 約 6 分 21 秒 (この PC。ファイル数が支配的) |
| 起動 → OBS が繋がる状態 (`/latest` が最初の JSON を返す) | 展開済み 2 回目 7.9 秒 / ビルド直後 8.5 秒 / **展開直後の初回 27.7 秒** |
| ランチャー単体 (設定検証 + MANIFEST 564 ファイル 18.5 MB 照合 + 版確認) | 1.2 秒 |
| バイトコード事前生成なし (初回コンパイル) | 150 秒で打ち切り (scipy.stats のコンパイル中、未完了) |

### 1.1 方式選定の根拠 (実測に基づくもの / 基づかないもの)

1. **サイズは torch が決める** (実測)。torch CPU wheel は 205 MB、展開すると 1,155 MB。うち
   `torch/lib/*.lib` (リンク用インポートライブラリ) が 700 MB 超 (`dnnl.lib` 653.5)。**実行時は不要なので削除**して
   351.8 MB。どの梱包方式でも torch の DLL (`torch_cpu.dll` 249 MB) は入るため、方式間の差は小さい。
2. **`python -m` で起動できる構造が spawn と相性が良い** (実測)。既存パイプラインは
   `mp.get_context('spawn')` で評価 worker を起こし (`scripts/run_live_pipeline_20260928.py:530`)、認識 worker も
   spawn する。埋め込み Python は `sys.executable -m ...` がそのまま効き、Windows で実際に完走した。
   凍結 (PyInstaller) は spawn 子の再 import・`freeze_support`・hidden import の調整が要る。
3. **学習済 joblib の版固定** (実測)。`exchange_event_v1〜v4` の joblib は scikit-learn 1.8.0 で保存。同梱の
   scikit-learn 1.8.0 で Windows 上でも全件読めて評価が動いた。凍結すると版固定が見えにくくなる。
4. **PyInstaller は今回試していない (未測定)**。onefile は毎起動に約 800 MB の展開が必要で不適と推定。
   onedir は方式 2/3 の調整コストと引き換えに得るものが小さいと判断した。**判断であって測定ではない。**
5. **ホストの Python 3.14.5 は実行時に使えない** (実測): torch 2.5.1 に cp314 wheel が無く、
   開発の版固定 (numpy 2.4.4 / scikit-learn 1.8.0 …) を再現するため 3.12 を同梱する。ホストの 3.14 は
   `pip download/install --target --platform win_amd64 --python-version 3.12` のビルド道具としてのみ使用。
   システム全体への導入は行っていない (pytest だけ `D:\puyo_analyzer\packaging\devsite` へ)。

### 1.2 さらに小さくする余地 (今回は未実施)

- **ONNX Runtime へ移行**: torch (352 MB) を 20〜50 MB 級へ。ただし CNN (`cnn_phase_b_large_v2.pt`) と
  `exchange_event_v1/M0/model.pt` の変換と、出力の同値検証 (bit-identical でない) が要る。**最大の削減余地**だが
  「認識結果を変える可能性」があるため設計合意後。
- ファイル数削減 (展開 6 分の主因): `site-packages` を zip import 可能な単一 zip へまとめる。scipy/numpy の拡張モジュール
  (`.pyd`) は zip から読めないため部分的。installer (Inno Setup 等) の方が展開は速い見込み (未測定)。
- `sympy` 39 MB / `pandas` 30.6 MB / `scipy` 86.8 MB: 閉包上は pandas・scipy が要る (sklearn→scipy、
  `visualize_advantage_overlay` が pandas を import)。削るには import 側の改修が要る。

## 2. 司書調査 (既存資産の所在)

### 2.1 入口と設定の優先順位

- 入口: `scripts/run_live_pipeline_20260928.py:523 main()`。`--source dshow|video`、`--config`、`--output`、`--video`、
  `--port` ほか。本番の撃ち合いイベント構成は `build_command` が `--production-exchange-event` を足して
  `production_config.exchange_event_flags()` から取得 (単一情報源)。
- 設定の優先順位 (`src/phase_j/live_config.py:13 apply_config`): **CLI 明示値 > `--config` の JSON > `config/live_defaults.json`
  (`cpu_threads=1, evaluation_nice=10, cnn_device=cpu`) > argparse 既定**。JSON の未知キーはエラー (許可外は
  `name/index/verification_only` のみ追加許容)。
- `dshow` では `--config` (機器の name/index を持つ JSON) が必須で、`realtime=True`・24 時間分の duration が入る。
- **機器は `index` で開く** (`src/phase_j/live_device.py:135` `VideoCapture(index, CAP_DSHOW)`)。`name` は
  較正ファイル名のハッシュ (`live_device.py:46`) に使う名札で、名前による選択はしない (3 段判別の①は index 指定)。
- 3 段判別の実装済み部分: ② 映像内容確認 = `PuyoScreenVerifier` (盤面枠 + 両側得点 OCR、`live_device.py:52`)、
  ③ 機器別較正 = `config/device_calibration/<sha256>.json` (実行時に保存)。
- 公開: `LiveOverlayServer` (`src/phase_j/live_publish.py`)。OBS ブラウザソース URL は **`http://127.0.0.1:<port>/`**、
  SSE は `/events`、最新は `/latest` (`src/stream_overlay.py:179`)。

### 2.2 実際に必要な Python 依存 (live 経路の import 閉包)

静的解析 (関数内 import 含む、`packaging/import_closure.py`): **内部 154 モジュール (src/scripts、ソース 4.3 MB)**。
外部: `numpy, torch, cv2, pandas, PIL, sklearn, joblib, jsonschema`、`puyo_core` (Rust、必須。下記の訂正)、`tests` (後述)。
matplotlib / statsmodels / lightgbm / yt-dlp / torchvision は **不要** (リポジトリ全体の requirements とは別物)。
開発 venv (CUDA 版 torch + nvidia-* = 約 4.3 GB) との比較で、配布物は CPU 版で 1/5 以下。

- **【訂正 2026-09-30 夕】`puyo_core` (Rust) は任意ではなく必須。** 初版は「HSV 分類の高速化のみで未ビルドでも黙って
  Python 経路に落ちる」(`production_config.py:683`) を根拠に同梱しなかったが、**本番構成は幽霊連鎖ルール ON**
  (`exclude_hidden_row_from_pop=True`) で、Python フォールバックはこれに未対応
  (`src/puyo_core_bridge.py:221` が `NotImplementedError`)。実測: puyo_core なしの配布版は、zenchi 265 秒付近
  (連鎖の先読み探索) で落ちた。初版の起動確認 (冒頭 30〜160 秒) はこの経路に到達しておらず、見逃した。
  → Windows 向けにビルドして同梱した (§11-1)。
  `tests` は `src/patch_classifier.py:501` の関数内フォールバックで、ライブ経路では到達しない。

### 2.3 必要な非コード資産 (実行時に open された実測)

`sys.addaudithook` で全プロセス (親 + spawn 子、5 process、総 open 2,359 回) を記録
(`packaging/audit_sitecustomize.py` / `audit_opens.py`)。ソース中の `data/ models/ config/` 文字列リテラルから
作った候補群について、**app 内で読まれた非コード資産 222 ファイル、同梱したのに未使用 0、app 内で存在しない open 0**。
(候補群は静的な文字列走査で作ったため、動的に組み立てるパスは候補から漏れうる。漏れは MISSING 検査で拾う設計。)

| 分類 | 内容 | サイズ |
|---|---|---|
| 認識 CNN | `models/cnn_phase_b_large_v2.pt` 0.4 MB、`cnn_global_best.pt`、`cnn_best.pt` | 0.4 MB |
| 撃ち合い評価モデル | `models/exchange_event_v1〜v4` (joblib + M0 `model.pt` + manifest) | 7.2 MB |
| 着弾カウンタ | `models/landing_counter_prob_v1/model.json` | ~0 |
| UI テンプレート | `models/ui_templates/**`、`next_pair_centroid_v1.npz`、`calibration_video01.json` | 1.1 MB |
| 色プロファイル | `data/per_video_hsv_ranges/**`、`data/puyo_profiles/**` | 0.9 MB (data 全体) |
| 較正・境界 | `data/indicators_v2/*calib*.json`、`data/verify/{chain_*,hidden_row_*}.json`、`match_boundaries_v5/*/matches.tsv` (4 件)、`retrain148_2026-08-14/*` | (data に含む) |
| スキーマ | `docs/schemas/puyo_overlay_{snapshot,health}_v1.schema.json`、`enums_v1.json` (`validator.py:24` が実行時に読む) | ~0.05 MB |
| 設定 | `config/live_defaults.json`、`live_evaluation.json`、`live_*.example.json` | ~0 |

**git 管理外**: `models/*.pt` と `data/` の大半 (`.gitignore`)。ビルドは `--asset-root` (既定は本体リポジトリ)
から補う。リポジトリ内 → asset-root の順に解決し、どちらにも無ければビルド失敗。

`data/verify/match_boundaries_v4` (125 MB) と `labeled_win_full148_*.csv` (553 MB) は **同梱しない**
(60 秒シナリオで open されず、候補からも外した)。ただし「別動画・別経路で必要になる」可能性は監査範囲外 (§8)。

## 3. 同梱物

```
PuyoLive/
  PuyoLive.bat               起動 (ダブルクリック)。--list-devices 等は引数で転送
  puyo_live.example.json     利用者設定の雛形 (利用者が puyo_live.json にコピー)
  README_ja.txt              利用者向け手順
  LICENSES/<配布名>/…        同梱ライブラリの LICENSE/NOTICE 44 ファイル
  python/                    埋め込み Python 3.12.3 + site-packages (26 パッケージ、版は MANIFEST に固定)
  app/                       src/ scripts/ (import 閉包のみ)  config/ models/ data/ docs/schemas/  MANIFEST.json
```

| 区分 | サイズ | 備考 |
|---|---|---|
| torch 2.5.1+cpu | 351.8 MB | `*.lib`/`include`/`tests` 削除後 |
| cv2 (opencv-python-headless 4.13.0.92) | 113.6 MB | FFmpeg DLL 同梱 (LGPL、§9) |
| scipy + scipy.libs | 86.8 + 20.3 MB | |
| sympy / pandas / scikit-learn | 39.2 / 30.6 / 28.5 MB | sympy は torch の依存 |
| numpy + numpy.libs | 17.0 + 21.0 MB | |
| app (models 8.7 / src 6.7 / scripts 2.1 / data 0.9) | 18.6 MB | pyc 含む |

バージョン固定: `packaging/requirements-live.txt` (直接依存 = 開発 venv と同版)、
`packaging/requirements-live.lock.txt` (今回実際に入った推移依存を含む 26 件)。

## 4. 利用者設定ファイル `puyo_live.json`

```json
{ "source": "dshow", "device_name": "OBS Virtual Camera", "device_index": 0, "port": 8765, "output_dir": "output" }
```

| キー | 既定 | 検証 |
|---|---|---|
| `source` | `"dshow"` | `dshow` / `video` (video は動作確認・デモ用) |
| `device_name` | (dshow で必須) | 空でない文字列。**選択には使わない名札** (較正保存名) |
| `device_index` | 0 | 0 以上の整数。**機器を選ぶのはこれ** |
| `video_path` | (video で必須) | 相対パスは設定ファイルの場所が基準 |
| `port` | 8765 | 1024〜65535 の整数 |
| `host` | `127.0.0.1` | 空でない文字列 |
| `output_dir` | `output` | 相対は設定ファイル基準。`pipeline_config.json` と実行記録を書く |
| `mc_rollouts` | 30 | 1 以上 |
| `start_sec`/`end_sec` | なし | video のみ。0 以上の秒 |

未知キー・型違い・範囲外は **項目名と直し方つきの日本語エラー** (終了コード 2)。JSON 構文エラーは行・列を出す。
CPU のみ (`cnn_device=cpu`)・`cpu_threads=1`・`evaluation_nice=10`・`coalesce_features=true` は固定
(既存 `config/live_dshow.example.json` と同一であることをテストで固定)。

## 5. ランチャー UX

`src/phase_j/launcher.py` (新規。既存パイプラインは無変更)。

1. 設定の検証 → 誤りなら `[設定エラー] …` + 終了コード **2**。
2. `MANIFEST.json` 照合 (存在・サイズ・SHA256 を app 全 564 ファイル、+ 26 パッケージの版) → 不一致は
   `[配布物エラー]` に種別 (欠落/サイズ/SHA256/版) と先頭 20 件 + 終了コード **3**。
   母数 0 の照合は成功と読み替えない (`ManifestReport.ok` は `checked > 0` が条件)。
3. `output_dir/pipeline_config.json` を書き、作業ディレクトリを app へ移して既存 `main()` を呼ぶ
   (既存は `config/ models/ data/` を相対パスで読むため)。
4. 起動時に `[起動] OBS ブラウザソース URL: http://127.0.0.1:8765/` を表示。
5. オプション: `--check-only` (検証だけ)、`--list-devices` (index 0〜8 の開閉・映像有無・解像度)、
   `--skip-manifest` (開発用)、`--require-manifest` (`.bat` が付ける)。

入力段階は既存実装が公開 DTO の `display.input_status/message` に出す:
`入力確認中` → (`ぷよ画面なし`) → `色を較正中 N%` → `ready`。今回の実測では、
**`http_ready` の時点で既に `入力確認中` が見えている** (OBS を先に繋いでおけば起動直後から表示が出る)。

## 6. OBS 側の手順 (利用者向け、`README_ja.txt` と同内容)

1. ボードの映像を「ソース出力」で仮想カメラに流す (配信用の重ね表示を含めない。1920×1080 / 16:9)。仮想カメラ開始。
2. `PuyoLive.bat --list-devices` で「映像あり 1920x1080」の index を確認 → `puyo_live.json` の `device_index`。
3. `PuyoLive.bat` を起動。
4. OBS に「ブラウザ」ソース: URL `http://127.0.0.1:8765/`、幅 1920 高さ 1080。
5. 「入力確認中」→「色を較正中」→ 判定表示。較正は機器別に保存され、次回は短い確認のみ。

## 7. 今回の実測で確認できたこと

- **Windows + 埋め込み Python + CPU torch で、既存パイプラインが最後まで動く**: 保存動画 (`v29_match2_156s.mp4`、
  `source=video`、realtime) を入力に、`入力確認中` → `ぷよ画面なし` → `色を較正中 1%…99%` → `ready` →
  評価 `available` まで遷移 (`/latest` の 0.1 秒ポーリングで観測、展開済み zip の複製から実行)。
  `first_available` は 30.7 秒 (2 回目)。この値は動画の冒頭 (ぷよ画面なし区間) と較正時間を含み、起動時間ではない。
- **実機 (DirectShow) 経路**: `source=dshow`、index 0 (この PC の 640×480 のカメラ。Web カメラか仮想カメラかは
  未特定) で `入力確認中` → `ぷよ画面なし`。**ぷよ画面でないカメラ入力を内容確認 (第 2 段) が弾く**ことを
  実デバイスで確認した (memory の Web カメラ誤選択リスクに対応する動作。ただし 1 デバイス・1 回のみ)。
  `--list-devices` はこの PC で index 0/1/2 = 640×480 / 720×480 / 640×480、3〜8 は開けないと表示。
  **キャプチャボード/OBS 仮想カメラの 1920×1080 入力は未検証** (この PC に接続なし)。
- 改ざん検出: `config/live_defaults.json` に 1 バイト追記 → `サイズ不一致 1 件` + 終了コード 3。復元で通る。設定なし → 終了コード 2。
- テスト: 新規 44 件 (`tests/test_launcher.py`、`test_launcher_manifest.py`、`test_packaging_closure.py`)。
  **配布用 Python (Windows 3.12.3) 上で既存 `tests/test_live_b6.py` (40 件) と合わせて 84 件通過**。
  ランチャーの出力設定を **実物の `parse_args` に通して意味 (機器入力・realtime・CPU・port) まで検査**している
  (写像が食い違うと落ちる)。WSL は本エージェントの環境から起動できず、WSL 側での実行は未確認。

## 8. 実測中に見つかった同梱漏れ・事故 (再発防止として道具に反映済み)

| # | 事象 | 原因 | 対処 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: src.phase_j.scheduler` | import 閉包が親パッケージ `__init__` 内の `from .x import` を辿らなかった | `import_closure.py` が親も閉包に含める + 回帰テスト |
| 2 | `FileNotFoundError: live_evaluation.py` | `asset_hashes()` が import されないソースも名前で読む | `src/phase_j` を丸ごと同梱 |
| 3 | `FileNotFoundError: docs/schemas/...schema.json` | `validator.py:24` が実行時に docs を読む | 3 ファイルを資産に追加 |
| 4 | 初回起動が 150 秒超 | バイトコード未生成 (scipy.stats 等のコンパイル) | ビルド時に `compileall --invalidation-mode unchecked-hash` |
| 5 | サイズ 1,603.9 MB | `torch/lib/*.lib` (リンク用) が大半 | `.lib`・`include`・`tests` を削除して 800.7 MB (削除後に zip 展開複製で再実行して動作確認) |

同型の再発を検知するため、`audit_opens.py` は「app 配下で open 試行されたのに存在しないパス」(`MISSING`) を出す。
**ただし監査は 60 秒の 1 シナリオ分**。試合終了・全消し・別テンプレートなど**通らなかった経路が読む資産は未確認**
(§9 の残作業 1)。

## 9. 範囲外・未了 (実リリース前に必要なもの)

> **2026-09-30 夕の更新**: 項目 1・3(Windows 内)・4・6・7・10(puyo_core) は §11 で対応/測定した。残りは §11-7 を参照。

1. **通し試験の被覆拡大** (最優先): 複数試合 (開始・終了・全消し・おじゃま・別ティア動画) を含む区間で MISSING=0 を確認。
   現状の監査は約 60 秒。`match_boundaries_v4` 等を読む経路が別にある可能性。→ §11-2 で実施 (全消しの通過は未確認)。
2. **実機 1920×1080 入力の通し試験**: OBS 仮想カメラ/キャプチャボードでの認識・較正 (OBS 変換で 1.08% セル誤読の B3 実測あり)。
   遅延・CPU 使用率も未測定 (他エージェントが遅延修正中の `live_snapshot.py` / `live_process.py` は本ブランチで無変更)。
3. **数値の同値確認**: 配布版 (Windows・CPU torch・numpy 2.4.4・`puyo_core` なし) と開発環境 (WSL) で
   同一動画の `display.npz` を比較していない。BLAS/oneDNN の差で微小差が出うる。
4. **クリーンな Windows での起動確認**: この PC は VC++ 再頒布可能パッケージ導入済みのため、未導入機での
   `torch_cpu.dll` ロード失敗は再現できていない。Windows Sandbox / 新規 VM で確認するか、
   `vcruntime140*.dll`/`msvcp140*.dll` の同梱を決める。
5. **インストーラ・署名**: zip 展開に 6 分かかる (ファイル数 13 千)。Inno Setup 等の導入、コード署名
   (SmartScreen 警告の回避)、自動更新は未着手。署名証明書の調達はユーザー判断。
6. **ライセンス**: 44 ファイルを自動収集したのみで、内容の法務確認と `NOTICE` 整理は未了。特に
   opencv-python-headless 同梱の FFmpeg DLL は LGPL、scipy.libs の OpenBLAS/GCC ランタイムは例外条項付き GPL を含む。
   学習データ (動画由来) のモデル再配布の可否、プロジェクト自身のライセンス表記もユーザー判断。
7. **書き込み先**: 機器別較正は `app/config/device_calibration/` へ書く (相対定数 `CALIBRATION_ROOT`)。
   Program Files 等では書けない。`%LOCALAPPDATA%` へ移すには `live_device.py` の改修が要る (本タスクでは触れていない)。
   `output/` の肥大化 (spool 等の上限) も未測定。
8. **配布物の再ビルド**: 他エージェントの遅延修正が rt へ入ったら、`python packaging/build_bundle.py --reuse-python --zip`
   (約 30 秒 + zip) で app だけ更新できる。**ビルドは現在の作業ツリーのソースを使う**ため、対象コミットを確認して行うこと。
9. **再現ビルド**: 依存取得 (`--download`) は `requirements-live.txt` から解決しており lock は使っていない。lock からの取得へ
   切り替える場合は torch のローカル版 (`+cpu`) の解決を要確認。
10. `puyo_core` (Rust) の Windows ビルド同梱 (認識 1.19 倍)、GPU 版・ONNX 化 (§1.2) は範囲外。
11. 出力 (`output/`) を配布版の起動ごとに掃除しない。長時間運転 (B15 1 時間運転相当) は配布版で未実施。

## 11. 2026-09-30 夕 第 2 段 (同値確認・被覆・VC++・較正保存先・ライセンス・ONNX)

指示: coordinator の追加依頼 5 項目 + ONNX (user 承認、条件は「品質が変わらないこと」)。数値の単位・条件を併記する。

### 11-1. puyo_core (Rust) の Windows 同梱 — 初版の見落としの是正

- 経緯: 同値確認のため zenchi 226〜340 秒を流したところ、265 秒付近で `NotImplementedError`
  (幽霊連鎖ルール ON に Python フォールバックが未対応、§2.2 の訂正参照) で落ちた。
- 対処: Windows 向け wheel を新規にビルドして同梱した。Rust 1.98.1 (windows-gnu)・maturin 1.15.0・zig 0.16.0 (リンカ)。
  ツールはすべて `D:\puyo_analyzer\packaging\` 配下 (rust / native_build / devtools)。`native/puyo_core` のソースは無改変
  (ビルド用コピー側の Cargo.toml だけ `pyo3` に `generate-import-lib` feature を足した。実行環境に Python 開発ファイルが無いため)。
  ビルド 5 分 17 秒、`puyo_core-0.1.0-cp312-cp312-win_amd64.whl`。.pyd の import は kernel32 / ntdll / bcryptprimitives / UCRT の API セット / python312.dll のみ (PE 走査。VC ランタイム・mingw 系 DLL への依存なし)。
- **検証の限界**: `tests/test_puyo_core_parity.py` は評価データ (`boards_lean_phase_l_*`) が worktree に無く 15 件すべて skip。
  Windows 版 native の正しさは §11-1 下の同値確認 (native を使うパイプライン全体の torch/ONNX 一致・run 間一致) と、
  クリーンな WSL 版との比較 (未実施、§11-7) に依存する。**native (Windows ビルド) と WSL の Linux ビルドの一致は未確認。**

### 11-2. 通し試験の被覆 (§9-1)

実行時に open されたファイルを、全プロセス (spawn した子を含む) で記録 (`packaging/run_coverage.py` + `audit_opens.py`)。

| シナリオ | 動画・区間 | 内容 | 所要 (6 並列・監査フック付き) |
|---|---|---|---|
| zenchi_m3_m4 | zenchi 226〜340 秒 | 試合 2 の終了 (228 秒)・試合 3 (231〜274)・次の試合開始・試合 4 途中 | 901 秒 |
| zenchi_m3_m4_onnx | 同上 (`PUYO_CNN_BACKEND=onnx`) | 同上 | 901 秒 |
| q_m2_m4 | q 590〜760 秒 | 試合 2・3・4 (3 試合連続) | 1,072 秒 |
| fcXG_m1_m2 | fcXG83vInDY 175〜385 秒 | 試合 1 の開始・終了・試合 2 | 1,306 秒 |
| mia8_m2_m3 | mia8KCjr52g 180〜350 秒 | 試合 2・3 | 1,019 秒 |
| v40_clip | evaluation_videos の v40 (別系統) 0〜120 秒 | 境界ファイルなし (ゲートなし) | 1,166 秒 |
| launcher_video (2 回) | v29 (`source=video`, realtime, `calibration_location=localappdata`) | 入力確認→較正→ready→評価 (161 秒で available) | 100 秒 + 240 秒 |
| launcher_dshow | 実機 index 0 (ぷよ画面ではないカメラ) | 入力確認→ぷよ画面なし | 60 秒 |

- **合算: 31 プロセス、総 open 2,556 件 (プロセスごとの重複除去後)、app 内で読まれた非コード資産 296 件。**
  **MANIFEST に無い物 0 件** (実行時生成の `config/device_calibration/` は除外する定義)。**app 内で存在しないパスへの open 0 件**。
  **同梱したのに一度も読まれなかった資産 0 件** (ONNX 同梱ビルドでの集計。全部がいずれかのシナリオで読まれた)。
- **モード依存に注意**: 非 realtime の CLI シナリオだけでは読まれた資産が 19 件しかなく、`data/per_video_hsv_ranges` や
  `data/puyo_profiles` 等 (計 277 件) はランチャー (lifecycle) 経路でだけ読まれた。**どちらか一方の被覆では足りない**ことが実測で分かった。
- **通っていない可能性のある経路**: 全消し (テロップ)・おじゃまの大量降下・試合の勝敗パネル (WIN★) 等は、区間に含まれたかを
  個別には確認していない (選び出さず、試合を含む区間を流しただけ)。資産の読込は初期化時にまとめて行われるため、
  読込漏れは検出できるが、その経路の**判定の正しさ**は本試験の対象外。
- 処理した動画時間は延べ約 1,000 秒 (CLI 6 本 = 114×2+170+210+170+120、ランチャー 90 秒)。

### 11-3. VC++ ランタイム (§9-4) — 依存関係の走査結果

`packaging/scan_dll_deps.py` (pefile で PE ヘッダの import テーブル + delay import を走査)。対象 320 ファイル (dll/pyd/exe)、
import される DLL は延べ 64 種。

| 分類 | 件数 | 内容 |
|---|---|---|
| 配布物内で解決 (importer の同居/アプリ ディレクトリ/パッケージ同梱 libs) | 21 | torch 系 DLL、libiomp5md、OpenBLAS、vcomp140 (sklearn/.libs) 等 |
| OS の API セット (`api-ms-win-*`) | 16 | UCRT。**Windows 10 以降の OS 標準** (Windows 10/11 のみ対象) |
| Windows 標準 (System32) | 27 | kernel32、user32、advapi32 等 |
| **VC++ ランタイムで配布物内に無い** | **1 (走査時点)** | **`msvcp140.dll`** (torch の `fbgemm.dll`、`functorch/_C.pyd`、`protoc.exe` ほか 6 ファイル) |

- `vcruntime140.dll` / `vcruntime140_1.dll` は埋め込み Python が同梱。`msvcp140.dll` は sklearn/.libs と numpy.libs にハッシュ付きの別名があるが、
  **torch からは見えない** (importer 別に判定しないと見落とす。初版の集計は名前だけで「同梱済み」と誤判定していた)。
- 対処: 再頒布可能な `msvcp140.dll` `msvcp140_1.dll` `msvcp140_2.dll` `concrt140.dll` を `python/` へ同梱
  (`build_bundle.py: VC_RUNTIME_DLLS`、取得元は System32)。走査の再実行で VC ランタイムの未解決は 0 になった。
- 起動前チェック (`src/phase_j/launcher_runtime.py`): `vcruntime140` / `vcruntime140_1` / `msvcp140` が python/ に無く OS からもロードできなければ、
  日本語で不足名と公式ダウンロード先 (`https://aka.ms/vs/17/release/vc_redist.x64.exe`) を出して終了コード 4。Windows 以外は対象外。
  **未導入 PC での実動作は再現できていない** (この PC は導入済み)。走査結果と単体試験 (注入した「ロード不可」で 3 DLL 全部不足を報告) で示した。
- 再頒布の可否: これらは Microsoft の再頒布可能コードに該当する DLL 群。条件の原文確認は未 (`docs/PHASE_J_PACKAGING_LICENSES_2026-09-30.md` §3-5)。

### 11-4. 機器別較正の保存先 (§9-7)

- `calibration_location`: `"app"` (既定、従来どおり `config/device_calibration`) / `"localappdata"` (`%LOCALAPPDATA%\PuyoLive\device_calibration`)。
- 変更は最小: `live_device.py` に環境変数 `PUYO_CALIBRATION_DIR` を 1 つ足しただけ (未設定なら従来のパスと同一。試験で固定)。
  ランチャーが設定から環境変数を設定し、spawn 子へ継承される。
- 実測: `localappdata` (LOCALAPPDATA を D: の検証用ディレクトリへ差し替えて) で v29 を流し、較正ファイルが
  `…\PuyoLive\device_calibration\b110fe….json` に保存され、`app/config/device_calibration/` は空のままだった。

### 11-5. ライセンス (§9-6)

`docs/PHASE_J_PACKAGING_LICENSES_2026-09-30.md` に一覧と配布条件を整理。要点: FFmpeg 4.4.6 (LGPL-2.1+、動的 DLL のまま同梱、
ソース入手先を README に記載)、GCC ランタイム (GPL+例外)、FreeType (FTL 謝辞)、MS 再頒布可能コード。
**モデルの再配布は user 承認済み (2026-09-30)**、自身のライセンス表記は user 判断待ち。

### 11-6. ONNX (CNN) — 合否結果

事前登録 `docs/PHASE_J_ONNX_PREREGISTRATION_2026-09-30.md` (評価前に固定)。既定 OFF、`PUYO_CNN_BACKEND=onnx` で切替、torch 版は残す。

**(a) CNN 全セル argmax 完全一致: 合格 (不一致 0 件)。**
- 母集団: 4 動画 x 各 1,200 フレーム (fcXG のみ重複除去で 1,199) = 4,799 フレーム、x 144 セル = **691,056 セル/モデル**、3 モデル
  (`cnn_phase_b_large_v2` / `cnn_global_best` / `cnn_best`) で **延べ 2,073,168 セル、argmax 不一致 0**。
- 確率の最大絶対差: 1.67e-6 (phase_b_large_v2) / 6.91e-6 (global_best) / 3.84e-6 (best)。
- **境界例** (torch の上位 2 クラス確率差): 差 < 1e-3: 49 / 29 / 34 件、< 1e-4: 3 / 5 / 1 件、**< 1e-5: 0 / 1 / 1 件**、< 1e-6: 0 件。
  境界例 (< 1e-5) は延べ 2,073,168 セル中 2 件 (別々のモデル)。これらも確率差が上位差を超えず反転しなかった。
  割れ得る領域は実在するが、今回の標本では反転 0 件。
- 注: mia8 は末尾 3 秒を除外 (最終フレーム付近が読めない実測。評価前の標本区間の機械的な補正であり、基準の変更ではない)。

**(b) 同一区間のパイプライン全体の一致: 合格 (差 0 件)。**
- 区間: zenchi 226〜340 秒 (試合 2 終了・試合 3・試合 4 の一部を含む)、非 realtime、配布版 Windows・CPU。
- 比較母数: `display.npz` 3,420 行、`settled.npz` 1,982 行、`events.jsonl` 5 行、`inputs.jsonl.gz` 9,143 行、
  `review_data.csv` 3,420 行 = **680,580 セル**。
- torch 版 2 回 (A: 旧コード、T2: 新コード) の差 = **0 件** (測定器の底が 0、かつ既定 OFF の bit-identical をパイプライン全体で確認)。
- torch 版 (T2) 対 ONNX 版 (O1、O2 の 2 回): **どちらも 0 件**。O2 は ONNX の使用実績を終了時ログで確認
  (`[cnn_onnx] sessions=2 calls=9551 patches=349743`。torch 版のログには 0 件)。
- 比較器の陽性対照 (差を入れたら必ず検出、行数差・欠落を不合格扱い) は `tests/test_compare_outputs.py` の 7 件で固定。
- 制約: `events.jsonl` は 5 行と少ない (原因は未調査)。イベント記録の一致の検定力は限定的。

**採用判断**: (a)(b) とも合格のため **不採用条件には該当しない**。ただし現状は torch を外せないため、サイズ削減の効果はまだ無い (下記)。

| 項目 | torch 版 (既定) | ONNX 同梱 (既定 OFF で切替可) | 差 |
|---|---|---|---|
| 展開後サイズ | 802.8 MB | 848.1 MB | +45.3 MB |
| zip サイズ | 271.0 MB | 287.0 MB | +16.0 MB |
| 起動→`/latest` 応答 (秒、3 回) | 5.3〜6.4 | 4.3〜4.8 (torch 経路) / 3.9〜5.3 (ONNX 経路) | 有意差なし (CNN 読込前に応答するため) |
| CNN 推論のみ (torch 1 スレッド、標本 1 フレームあたり 144 パッチのバッチ、400 フレーム、ms/フレーム) | phase_b 22.8 / global_best 12.6 / best 16.4 | 17.5 / 11.8 / 15.4 | −23% / −6% / −6% |
| 同 (1 パッチあたり ms) | 0.158 / 0.087 / 0.114 | 0.122 / 0.082 / 0.107 | 同 |
| パイプライン 1 フレームの「読出し開始→認識完了」P50 (ms、3,420 フレーム、torch と ONNX を同時並走) | 139.3 | 131.7 | −5.5% |

- **CNN 時間の割り算**: 各モデルの総 `predict_proba_batch` 壁時計 ÷ 標本フレーム数 (144 パッチ/フレーム) と ÷ パッチ数。デコード・前処理 (パッチ切出し) は含まない。
  torch は `set_num_threads(1)` (本番の cpu_threads=1 と同条件)、ONNX は `intra_op=1`。閑散時の測定。
  (スレッド数を揃えない初回測定 (torch 既定スレッド) では torch のほうが速く出た。条件を揃えて再測定した値を採る。)
- パイプライン P50 は 2 本並走・他ジョブ稼働中のため**目安**。認識 1 フレーム = 動画 1 フレーム (30fps 正規化) あたり。
- **torch を外した場合の見込み (推定、未測定)**: torch 352 MB + sympy 39 MB + networkx 7 MB 等の依存を除いて展開後 約 430 MB 前後。
  **現状は外せない**: `recognition_pipeline.py` の `torch.load` / `nn.Module` 構築 (ONNX の重みハッシュ算出も torch のテンソルを使う)、
  `exchange_event_m0.py` (M0 の `model.pt`)、`state_pipeline.py`、`next_pair_classifier.py`、`ojama_cnn.py`、
  `phase_j/live_cpu.py`・`live_process.py` の `import torch` が残る。M0 の ONNX 化・torch なしの重み読込・`live_process.py` の改修
  (別エージェントが編集中) が要る。
- **重要 (別件の実測)**: Windows・1 スレッドでの認識は 1 フレーム P50 約 131〜141 ms。30Hz の予算 (33 ms) の 4 倍で、
  realtime 入力では取りこぼしが出る前提になる。遅延修正 (別エージェント) の設計前提と突き合わせること。

### 11-7. 未了・依頼事項

1. **WSL 版との比較 (同値確認の本命) は未実施。** 本エージェントの環境では `wsl` コマンドが実行できなかった
   (隔離ハーネスが拒否)。Windows 内の比較 (torch 対 torch、torch 対 ONNX) は完了。
   **依頼**: `packaging/run_equiv.sh` を WSL で実行 (配布版 `app/` をそのまま使うため、ソース・資産・CLI が Windows 側と同一になる):
   `wsl -d Ubuntu -- bash /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/.claude/worktrees/agent-ac71f2c4855be6bef/packaging/run_equiv.sh wslA /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/data/frames/video_zenchi_c0BQoMJwwQU.mp4 226 340`
   (約 10〜20 分)。その後 Windows 側で
   `python packaging/compare_outputs.py D:\puyo_analyzer\packaging\equiv\winT2 D:\puyo_analyzer\packaging\equiv\wslA`。
   **注意**: `winT2` は現ビルドの torch 版と同一コード。ビルドを更新した場合は再度 T2 相当を取り直すこと。
   WSL は puyo_core が Linux ビルド、torch は cu121 (CPU 強制)、Windows は windows-gnu ビルド + CPU torch。差が出たら分類: (i) torch/oneDNN の数値差、
   (ii) puyo_core (Rust) の OS 間差、(iii) パス区切りなど無関係な文字列 (比較器が `path_separator_only` として別計上)。
2. Windows 版 puyo_core と WSL 版の**直接の同値確認** (`tests/test_puyo_core_parity.py` を評価データありの環境で)。
3. 実機 1920×1080 入力、クリーンな Windows 起動、インストーラ・署名、`output/` 肥大・長時間運転、全消し等の通過確認は §9 のとおり未了。
4. `torch` を外す作業 (§11-6) は M0 の ONNX 化等が要り、別エージェントが触るファイルにも及ぶため、着手前に調整が要る。

## 10. ファイル一覧 (本ブランチで追加)

- `src/phase_j/launcher.py`、`src/phase_j/launcher_manifest.py`
- `packaging/build_bundle.py`、`import_closure.py`、`bundle_assets.txt`、`requirements-live.txt`、`requirements-live.lock.txt`、
  `PuyoLive.bat`、`puyo_live.example.json`、`README_ja.txt`、`smoke_launch.py`、`audit_sitecustomize.py`、`audit_opens.py`、`bundle_sizes.py`
- `tests/test_launcher.py`、`tests/test_launcher_manifest.py`、`tests/test_packaging_closure.py`
- 出力 (git 管理外): `D:\puyo_analyzer\packaging\build\PuyoLive\` と `PuyoLive.zip`、`downloads\`、`test\`

再現手順:
```
python packaging/build_bundle.py --download      # 埋め込み Python と wheel を D: へ (初回のみ)
python packaging/build_bundle.py --zip           # 組み立て + zip (初回 約 12 分)
python packaging/smoke_launch.py <PuyoLive> <puyo_live.json>   # 起動段階の実測
```

## 12. 2026-09-30 夜 FFmpeg (LGPL) と動画入力の除去、ライセンス MIT

user 決定 (2026-09-30): 配布版から動画ファイル入力 (`source=video`) と同梱の FFmpeg を外す。ツール自身は MIT。

### 12-1. 変更

- 配布物から `cv2/opencv_videoio_ffmpeg4130_64.dll` を除去 (`build_bundle.py: FFMPEG_GLOB`)。配布物内の `ffmpeg` / `avcodec` を名前に持つファイルは **0 件**
  (他の FFmpeg 由来は無い: `cv2.pyd` に FFmpeg コードなし、Pillow の `_avif` は libavif/aom で FFmpeg ではない)。
- ランチャーは `source=video` を受け付けず、終了コード 2 と
  「配布版は OBS 仮想カメラ等の入力 (DirectShow) のみ対応です。動画ファイル入力 (source="video") は使えません。…」を出す。
  開発用の `--dev-allow-video` (CLI のみ、設定ファイルには無い) は残る。`src` の既存の動画入力経路は無変更 (WSL の開発環境はそのまま)。
- `config/live_video.example.json` を配布物から除外。`LICENSE` (MIT) をリポジトリ直下・配布物直下・`app/LICENSE` (MANIFEST 対象) に置いた。
- README_ja.txt: MIT の日本語要約 (正式条文は英語の LICENSE)、モデルも同条件、非公式・セガとは無関係・登録商標の表記、無保証。

### 12-2. サイズの変化 (ONNX なしの既定ビルド同士、バイト数から換算)

| | FFmpeg あり (第 2 段末) | FFmpeg なし | 差 |
|---|---|---|---|
| 展開後 | 802,810,581 B (802.8 MB) | 774,236,330 B (774.2 MB) | −28.6 MB |
| zip | 270,993,649 B (271.0 MB) | 258,826,465 B (258.8 MB) | −12.2 MB |

ONNX 同梱ビルドは今回再ビルドしていない (差は同程度の見込み、推定)。

### 12-3. DirectShow 入力が FFmpeg なしで動くことの確認

- **PE 依存走査** (`scan_dll_deps.py`、DLL 除去後 319 ファイル・import 59 種): FFmpeg 系 (ffmpeg / avcodec / avformat / avutil / swscale / swresample) の
  import は **0 件**。`cv2.pyd` は FFmpeg を import テーブルに持たず、`opencv_videoio_*` プラグイン名を文字列として持ち実行時に探す方式
  (`cv2.getBuildInformation()` の「FFMPEG: YES (prebuilt binaries)」はそのプラグインの意味。DirectShow は「YES」で組込み)。
  **注**: PE 走査だけでは実行時ロードは分からないので、次の実行で補った。
- **実行 (1) 動画は開けなくなった**: 除去後、`cv2.VideoCapture('….mp4').isOpened()` は False (期待どおり)。
- **実行 (2) DirectShow の列挙・開閉・取得は動く**: `--list-devices` で index 0〜4 が「映像あり」(640x480 / 720x480 等)、5〜8 は開けない。
  `CAP_DSHOW` + 1920x1080 指定では 幅1920 x 高さ1080 のフレームを 5 デバイス中 4 つで取得 (残り 1 つは 1280x720) (Web カメラ・NVIDIA Broadcast 等)。
- **実行 (3) OBS 仮想カメラ経由の通し**: この PC に OBS Studio が入っていた。専用のシーン (v29 の保存動画をループ再生する
  メディアソース、1920x1080) を一時的に作り、`--startvirtualcam` で仮想カメラに流した。OBS Virtual Camera は index 4 (1920x1080、
  ぷよぷよ画面が映ることを画像で確認)。FFmpeg なしの配布版で `source=dshow`、`device_index=4` を起動:
  `入力確認中` (起動 15.0 秒) → `ぷよ画面なし` (25.5 秒、動画の冒頭) → `色を較正中 30%` (29.6 秒) → `ready` (31.1 秒) →
  **評価 `available` (33.2 秒)**。実行後に OBS を終了し、作った一時シーンを削除、`global.ini` / `user.ini` を退避コピーから復元した
  (利用者の既存シーンコレクション「無題」は未変更)。
- 制約: OBS の出力は動画のデコード結果で、実キャプチャボードの色・遅延特性とは異なる。1 回の実行で、長時間運転・遅延は未測定。

### 12-4. 残り

- §9 の未了項目 (WSL 版との同値確認、クリーンな Windows、インストーラ・署名 等) は変わらない。
- 動画入力を使う開発用ツール (`smoke_launch.py` の video 設定、`run_coverage.py`、`verify_onnx_parity.py`) は、**FFmpeg のある環境**
  (開発用に退避した `D:\puyo_analyzer\packaging\dev_ffmpeg\opencv_videoio_ffmpeg4130_64.dll` を cv2 フォルダへ戻す、または WSL) でだけ動く。
  配布版の検査は dshow で行う。
- 過去の実測 (§7、§11) の動画入力による数値は、FFmpeg あり配布物での結果であり、除去後に取り直してはいない
  (認識・評価のコードは無変更で、デコード経路のみが異なる)。
