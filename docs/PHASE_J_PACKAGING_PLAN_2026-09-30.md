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
外部: `numpy, torch, cv2, pandas, PIL, sklearn, joblib, jsonschema`、任意 `puyo_core` (Rust)、`tests` (後述)。
matplotlib / statsmodels / lightgbm / yt-dlp / torchvision は **不要** (リポジトリ全体の requirements とは別物)。
開発 venv (CUDA 版 torch + nvidia-* = 約 4.3 GB) との比較で、配布物は CPU 版で 1/5 以下。

- `puyo_core` (Rust): 認識の HSV 分類の高速化 (1.19 倍、認識結果は bit-identical。`production_config.py:683`)。
  未ビルド環境では黙って Python 経路に落ちる。**配布物には入れない** (Windows 向け wheel の用意が別作業)。
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

1. **通し試験の被覆拡大** (最優先): 複数試合 (開始・終了・全消し・おじゃま・別ティア動画) を含む区間で MISSING=0 を確認。
   現状の監査は約 60 秒。`match_boundaries_v4` 等を読む経路が別にある可能性。
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
