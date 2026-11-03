# ONNX バックエンド 事前登録 (2026-09-30、評価実行の前に固定)

承認: user が「品質が変わらないこと」を条件に ONNX 化を承認 (2026-09-30)。既定 OFF、torch 版は残す。
**以下の合否基準・母集団・設定は評価前に固定し、結果を見てから変えない。**

## 実装 (既定 OFF)

- 切替: 環境変数 `PUYO_CNN_BACKEND=onnx`。未設定なら従来どおり (`src/patch_classifier.py` の `_forward` は
  `self._model(batch)` をそのまま呼ぶ = bit-identical。試験で固定)。
- ONNX は重みの内容ハッシュ (`models/onnx/<digest>.onnx`) で引く。対応が無ければ例外 (黙って torch に戻さない)。
- 差し替え範囲: `CnnPatchClassifier` / `CnnPatchClassifierLarge` の畳み込み本体の logits のみ。前処理・softmax は既存のまま。
  **M0 (`exchange_event_v1/M0/model.pt`) と次ツモ検出など他の torch 利用は本作業の範囲外 (torch のまま)。**
- ONNX Runtime: 1.30.0、CPU、`intra_op_num_threads=1`、グラフ最適化は**既定 (ORT_ENABLE_ALL) 固定**。
  設定の調整は行わない (評価後の設定探索を禁ずる)。書き出し: opset 17、torch 2.5.1+cpu の TorchScript エクスポータ。

## 合否基準

**(a) 全セル argmax 完全一致**
- 母集団: 4 動画 (q / zenchi 2580〜3427 秒 / fcXG83vInDY / mia8KCjr52g、`D:\puyo_analyzer\videos\source` 3 本 +
  zenchi は `data/frames/video_zenchi_c0BQoMJwwQU.mp4`)。各動画 **1,000 フレーム以上**を、試合区間
  (`data/verify/match_boundaries_v5/*/matches.tsv` の各試合の開始〜終了、前後 2 秒を含む) の上で**等間隔**に標本化する
  (試合開始・連鎖・おじゃま・試合終了を含むかは、密な等間隔標本に任せる。特定の場面を選び出さない)。
- 単位: 標本フレームごとに 1P/2P の可視 12x6 = 144 セルの実パッチ (`predict_proba_grid` と同じ切り出し)。
- 対象モデル: `cnn_phase_b_large_v2.pt` / `cnn_global_best.pt` / `cnn_best.pt` の 3 つ全部。
- **合格 = argmax 不一致セルが全動画・全モデルで 0 件**。確率の最大絶対差と母数 (フレーム数・セル数) を併記する。
  上位 2 クラスの確率差が 1e-3 / 1e-4 / 1e-5 / 1e-6 未満の境界例の件数も併記する (割れ得る領域の大きさの目安)。

**(b) 同一区間のパイプライン全体の一致**
- 同一の保存動画区間 (zenchi 試合 3〜4 = 226〜340 秒) を、配布版パイプラインで torch 版と ONNX 版の 2 回流す
  (どちらも同一の app・同一の Windows 環境・非 realtime・`--no-split-evaluation --no-async-counter`)。
- 比較 (`packaging/compare_outputs.py`): `display.npz` / `settled.npz` (盤面・勝率列)、`events.jsonl`、`inputs.jsonl.gz`、
  `review_data.csv` を行 (セル) 単位で厳密比較。**合格 = 全ファイルで差 0 件** (母数併記、比較対象の欠落があれば未確認 = 不合格扱い)。
- 先に torch 版を 2 回流して run-to-run の差 (測定器の底) が 0 であることを確認する。底が 0 でなければ (b) は判定不能とし、報告する。

## 不合格時

1 件でも差が出たら**不採用**。torch 同梱のままとし、差の分類 (境界例か、実装の取り違えか) を報告するに留める。
再挑戦 (設定変更や別実装) は user 判断を仰ぐ。

## 合格時の報告項目

- サイズ (展開後・zip): torch 版と、(torch を外せる場合の) ONNX 版。**現状は M0 等が torch を使うため torch を外せない。**
  外せない場合は「ONNX Runtime を追加しただけ」の実測サイズと、torch を外せた場合の見込み (推定と明記) を分けて報告する。
- 起動時間 (秒、`/latest` 初回応答まで)。
- 認識 1 フレームあたりの時間: **何で割ったかを明記** (CNN 推論のみ = 標本フレーム 1 枚あたり 144 パッチのバッチ推論の壁時計 ÷ フレーム数、
  torch と ONNX それぞれ。パイプライン全体の 1 フレームあたりは別掲)。
- numpy 等の数値差で argmax が割れる境界例: 件数と例。
