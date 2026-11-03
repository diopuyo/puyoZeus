# 式枠自動合わせの製品採用（2026-10-17）

## 本線と取り込み

製品本線は `claude/live-row1-guard-20261014`、作業先は
`<work>/wt_live_row1guard`。取り込み前HEADは
`a6a0011e117d57ef798726e5237fd78898efa947` でoriginの同名ブランチと一致した。
前回 `D:/puyo_analyzer/packaging/build_20261016a/verification.json` のcode_sha、
`BUILD_NOTES.txt` と `build_recipe.py` もこのブランチ・コミットを記録している。
row1本番ONの履歴は `d8ba918f`、続く試験独立化が `a6a0011e`。

`claude/live-score-roi-align-20261016` の `4b590092` までを
`git merge --ff-only` で本線へ取り込んだ。直系子孫のため競合なし。
pushはPC1 Claudeの確認後とするユーザー指示に従い実施しない。

## 採用と設定経路

単一情報源は `src/production_config.py` の
`LIVE_SCORE_ROI_ALIGNMENT_DEFAULT=True` と `LIVE_SCORE_ROI_ALIGNMENT_DELIVERY`。
user 10/16決定「既定OFFで作り、試験合格後に製品ON」、PC2 eo2全5本PASS、
`done/eo2_live_roi_align_gate/INDEPENDENT_REVIEW_PC2CLAUDE.md` の独立検収合格を根拠とする。
39/94は有効式0.51→16.07%・5.56→17.22%、(0,-2)px。
29/65/79は(0,0)pxで、認識・公開勝率・死亡・個数DTO・式・得点原票が完全一致。
settled.npzの事後EMAの既知非決定性は公開表示と区別する。

CLI・ランチャーはこの定数を参照し、配布設定にも `align_score_roi: true` を明記する。
既定ONの項目は明示falseも子CLIへ配送するようにした。従来の「trueだけをJSONへ書く」
処理では、falseが省略され子CLIの既定ONへ戻るため。この共通処理は既定ONのrow1にも
適用されるが、既定動作は不変。`--no-align-score-roi` とJSONのfalseは引き続き利用可能。
低水準RecognitionPipeline APIの既定Falseはrow1と同じく維持し、製品設定でONを配送する。
NCC閾値0.80・式合格線0.70・B3モデル・重み・評価処理は変更しない。

## 検証とビルド

前回同様、既存埋め込みPythonを新しいC:ステージへ複製し、
`packaging/build_bundle.py` の `build(reuse_python=True, with_onnx=True)` を使用。
前回ビルドのB3モデル・時間表・ONNXを再利用し、モデルのSHA256一致とMANIFESTを検査する。
署名なしZIPと展開済み配布物の出力先は
`D:/puyo_analyzer/packaging/build_20261017a/`。既存ビルドは上書きしない。
ビルド台帳・設定検査・SHA256はこの出力先に保存する。

既定ON対明示OFFの合成入力試験は180秒・5,400コマで全出力差0。
探索中2,046コマ／確定後3,354コマ、有効式1,800側コマ、確定60段。
盤面readerとモデル値はテスト用固定であり、実映像の品質根拠はPC2 eo2と独立検収。
製品設定の未指定ON、明示OFF、JSON→CLI→認識workerの配送を単体でも検査した。

広範囲の関連試験は変更前 `4b590092` と同じ環境で比較した。
共通の既存失敗はfixtureのevaluation_result不足14件、OBS模擬接続1件、
他試験の環境変数残留2件、models/calibration_video01.json不足2件。
新規ONの環境変数をrow1単独検査へ持ち込む1件は試験の隔離を修正し、
ON環境からのrow1試験49件がすべて通過した。
Playwright・websockets・Chromium・libasoundは一時領域に用意し、ブラウザ試験を再確認する。
最終集計と各失敗の変更前後照合は配布先 `test_summary.json`、原票は `tests/` に保存する。
既存の環境・fixture・表示試験の不具合を本変更へ混ぜて修正しない。
