# 公開前のゲーム画像依存の調査（2026-10-07）

公開・配布不可。画像非同梱の置換は得点 OCR 候補の検証までで、全体の合格ではない。
現行の配布フォルダは変更していない。wt_b3_rt の変更は実験経路を既定 OFF で追加したもの。

## 読込箇所と用途

`scripts/audit_public_runtime.py` が src と scripts の画像読込を AST から抽出する。
結果は `D:/puyo_analyzer/b3_data/codex_tasks/signing_repo_prep/runtime_audit.json`。
配布の3入口の静的 import 閉包は調査時252モジュール。静的閉包には呼ばれない分岐も含む。
動的な生成・関数呼出の到達性は、これだけでは証明できない。

- `src/score_ocr.py`: `ScoreOcr._load_templates_from_dir` が `score_digits/digit_0.png`～`digit_9.png`。
  `ScoreOcr.load_default` が `score_digits/formula_mult.png`。得点8桁と連鎖式の数字・掛け算記号。
  10数字＋記号1枚で11枚である。学習器の試作はこの経路だけを置換済み。
  追加の利用先として `recognition_pipeline` が `_mult_template_gray` を
  `PlacementSignalRuntime` に渡し、`placement_signal_images.multiply` が余白付きの広いROIで
  掛け算記号を探索する。これは通常の式OCRとは異なる入力範囲であり、候補では未対応。
  R1置き照合と同時有効化すると明示的な例外で停止し、黙って合図を失わない。
- `src/score_template_ocr.py`: `load_default_templates` が同じ数字10枚。
  調査時の静的閉包外。別の OCR API なので、公開側で残すなら別途置換または削除が必要。
- `src/ui_mask.py`: `UiMaskMatcher.load_default` が `x_mark*.png` 6枚。
  窒息列の×印を赤ぷよと誤認しないための空セルマスク。小型分類器への置換が必要、未着手。
- `src/match_end_detector.py`: `MatchEndDetector.load_default` が `match_end_batan.png` と `match_end_yatta.png`。
  試合終了検出とロックダウン用。盤面推論の凍結に影響する。左右別ロゴ分類器の候補が必要、未着手。
- `src/review_terminal_outcome.py`: `_load_templates` が同じ勝敗2枚。
  やったロゴのみの追加 crop と、左右の勝者・敗者の矛盾検査がある。
  ファイル読込を消すだけでは死亡確定契約を維持できない。単純な2値検出で代替しない。
- `src/exchange_event_terminal.py`: `ObservedDeathDetector.__init__` が `match_end_batan.png` をグレーで読む。
  死亡の観測証拠用。上記と併せて置換・到達性の検証が必要。
- `src/telop_detector.py`: `TelopDetector.load_default` が `telop_challenger.png` と
  `telop_challenger_decisive.png`。大会の中央テロップの bbox と被覆セルを検出する。
  **全消しテロップではない**。分類だけでなく bbox の互換性も必要。未置換。
- `src/score_zero.py`: `ScoreZeroDetector.load_default` が `score_zero/zero_1P.png` と `zero_2P.png`。
  スコアゼロの補助検出。現行配布ランチャーでは score_zero_from_ocr=True により OCR を使用するが、
  公開側にこの旧ローダーを残さない整理が必要。
- `src/chain_count_ocr.py`: `ChainCountOcr._load_template_sources_from_dir` が
  `chain_count_digits/digit_N.png` と追加ソースを読む。連鎖数表示の OCR。
  既存の配布資産コメントでは呼出無しとされている。B3の式カウントとの関係を含め再確認が必要。
- `src/ojama_warning.py`: `_load_templates` が `ojama/{small,rock,line,moon,crown,big_crown}.png` と
  同名 prefix の追加画像。CNN が低信頼度のときの NCC 補助。
  **B3の `ObservationCapture` が必須としており、build_bundle.copy_b3_assets が通常の資産一覧の外で追加する。**
  CNNのみへの変更では低信頼度時の結果が変わるため、代替と比較が必要。未置換。
- `src/win_panel.py`: `WinPanelDetector.load_default` が `win_panel/star_win_star.png`。
  結果パネル用。現在の実行時到達性の確認、除外または分類器への置換が必要。
- `src/phase_j/distribution_tray.py`: `start_tray` が `brand/app_logo.ico`。
  所有者提供のブランド画像でありゲーム画面ではない。新 repo では画像を一切コピーせず、
  公開ライセンスと由来の確認後に独立して追加する扱い。

`src/calibration.py` の reference.png は利用例、`src/cli.py` の画像読込は利用者入力。
研究・学習・分析スクリプトの画像読込も原票には列挙するが、公開配布にそのコードを入れない。
`src/all_clear_detector.py` の全消し判定は盤面と得点を使い、画像テンプレートの読込はない。
認識 pipeline の全消し持越し状態と確定盤面の一致は、全体比較で別に確認する必要がある。

## 得点 OCR 候補

`src/image_free_digit.py` は小さな ExtraTrees の分類・回帰器を使う。
切り抜きの画素配列を形式変更して配る方式ではない。実行時の PNG 読込・旧画像への fallback はない。
モデルには決定木の分岐と葉の値を保存する。未学習・未配置のモデルは例外で止まる。
採用フラグ `IMAGE_FREE_SCORE_OCR_DEFAULT=False` は `src/production_config.py` が情報源。
実験時だけ `PUYO_IMAGE_FREE_SCORE_OCR=1` と `PUYO_IMAGE_FREE_DIGIT_MODEL` を指定する。

比較器は旧 OCR の決定を教師にする。正解アノテーションに対する絶対精度ではなく、旧方式との一致率。
動画の時間順の前60%から144フレームを学習に使い、後40%の96フレームで比較した。
v1は数字1528/1536・得点190/192・式192/192一致。
3099.610879秒の両側で明るさ変化により得点が欠測になった。
v2は濃淡の平均・標準偏差を正規化して数字1536/1536・得点192/192・式192/192一致。
旧方式で読めた得点は153件、有効な式は32件。欠測同士だけを数えた試験ではない。
ただし全フレームでも別動画でもなく、勝敗・中央テロップ・×印・予告・確定盤面・B3出力は未比較。
v1を見て変更した後の同じ標本なので、独立した最終検証と扱わない。

追加の密な比較（c68後半、約1.3秒おき960フレーム）では、数字15346/15360
（99.9089%）、得点1920/1920（100%）、式1919/1920（99.9479%）一致。
旧方式の有効な得点1531件・有効な式310件に対し、候補の有効な式は309件。
数字の不一致場面は11件。2119.291360秒の2Pで「40×64」の「4」が欠測となり、
式が `left_implausible` で棄却された（旧方式のその段の素点2560）。
他の差は両方式とも最終得点・式が欠測の場面だが、個々の桁の返り値は異なる。
原因は旧NCCと決定木の分類境界の不一致。段の連続確認が救済するか、
B3特徴や勝率に差を伝えるかは連続フレームのpipeline比較未実施のため未確定。
**この追加比較は不合格。v2重みを公開版へ採用しない。**

実験・不一致一覧・重みは `D:/puyo_analyzer/b3_data/codex_tasks/signing_repo_prep/` の
`digit_trial`、`digit_trial_v2`、より密な追加比較 `digit_dense_v2` に置く。画像は保存しない。
一時停止の確認は `scripts/obs_priority_guard.py` が20秒間隔のチェックポイントで行う。
分類器学習は1スレッド・1本ずつの木で区切る。

## 公開準備の制限

新規 repo は `D:/puyo_analyzer/puyozeus-live`。文書・モデル・Rust・組み立ての準備だけを入れる。
研究コードを含む252モジュールの閉包を丸ごとコピーしない。
画像依存の全置換が合格してから、推論専用コードを移行する。
`b3_train_model` 内の型を参照するモデル互換読込、研究用関数との切り分けも未完了。
新 zip の組み立て・「起動 → 初期設定 → 試合待ち」試験は未実施。
公開ゲートを false に維持し、push と GitHub repo 作成は一切行わない。

## 継続作業: image_free_recog（2026-10-07）

前回の既定OFF試作を `1a8cbb11` にコミットした。追加学習候補v3は既知の11場面の
不一致を解消し、2119.291秒2Pの40×64を復元した。ただし既知場面とその前後を
学習に追加した回帰試験であり、独立評価の合格ではない。

`scripts/compare_image_free_digit_stream.py` はc68の先頭から188,400フレームを
順に読み、得点・式・各桁を比較する。時刻指定の間引きは行わない。途中経過では
121.817秒2Pの旧得点0→新欠測など、新しい差を確認した。信頼度も差分に記録する。
旧方式も候補もフレームは同一の復号結果を使う。モデルv3は試験中に更新しない。
これはOCR単体の比較であり、確定盤面・B3特徴・表示勝率への影響を示すものではない。
得点欠測が勝率に影響しないと断定せず、重要度は未確定とする。

R1広域×は `placement_multiply_reader` を通じて差し替える。旧方式は従来の
領域・テンプレートをそのまま利用し、候補は専用の学習器を使う。専用学習器が
ないv3以前の候補は、R1初期化時に例外となる。v4の広域回帰器は未採用。
窒息位置の×印にも画像を読まない分類器候補を追加した。いずれも既定OFF。
勝敗ロゴ・中央テロップ・予告などの残りの画像依存は未置換。

原票・候補重みは `D:/puyo_analyzer/b3_data/codex_tasks/image_free_recog/`。
`c68_stream_v3_batch/comparison.json` は途中経過/完走の区別とフレーム総数を記録し、
`differences.jsonl.gz` は全ての不一致側をフレーム番号・時刻付きで保存する。
既知回帰試験は `digit_v3/`、広域候補は `digit_v4/`、×印候補は `ui_mask_v1/`。
候補・検証器は20秒間隔でPython版と配布exe版のPuyoLive稼働を確認して待機する。
旧 `c68_stream_v3/` は高速化前の中断原票。候補v3を変更せず、24フレーム分を
まとめる推論に切り替え、先頭から再比較した。実フレーム24枚・両側48件で
単フレーム推論との返り値（信頼度と式を含む）の完全一致を確認した。
後半の×印標本は1261/1280一致（旧発火522、新発火533）で不合格。
R1広域×標本は1919/1920一致で不合格。2479.438秒2Pで旧スコア0.83156に対し
候補0.63480となり、0.70の検出閾値を下回る見落としが残る。

zenchi原動画は既知の保存先とDドライブの探索で見つからなかった。通常取得と
Node指定の取得はともにHTTP403で失敗。取得ログは原票ディレクトリ、途中データと
保持契約は `D:/puyo_analyzer/local_frames/image_free_recog/source/` に置いた。
2026-08-31の元動画保持契約に従い削除しない。yt-dlpの実行系設定は
[公式EJS手順](https://github.com/yt-dlp/yt-dlp/wiki/EJS)を参照した。

新repoへの移行、画像0件の最終点検、zip作成と起動試験の条件は未成立。
新repoと現行PuyoLive・配布フォルダは変更しない。
