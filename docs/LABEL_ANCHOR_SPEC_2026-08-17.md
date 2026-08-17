# ラベル規約v2: 内容ベースアンカー (W8根治、2026-08-17)

## 背景 (測定器事故、`docs/KNOWN_WEAKNESSES.md` W8)

2026-07-31 に作った物差しラベル (52盤面3744セル) は `video_id + frame_idx`
という**絶対フレーム番号**で元動画に紐づいていた。元動画はストレージ節約
ポリシーで削除済みで、2026-08-14 に再DLした動画は解像度・fps・カット位置が
異なり (同一動画IDでも内容が異なる、`feedback_redownload_content_drift_
2026-08-14.md`)、同じ `frame_idx` が同じ画面を指さなくなった。結果、
52盤面中46盤面 (88.5%) がアンカー不能になった。

**根治**: ラベルの主キーを「参照画像そのもの (ハッシュ+周辺フレーム
シグネチャ)」にし、`frame_idx` は補助キーへ降格する。動画が再DLされても
参照画像で再アンカリングできるようにする。

## 新規ラベル1件の必須フィールド

| フィールド | 種別 | 説明 |
|---|---|---|
| `sheet` | ファイル名 | ラベルPNGのファイル名 (labels.tsv の行キー) |
| `anchor_hash` | **主キー** | 対象フレームの盤面クロップの平均ハッシュ (16進、`_label_anchor_lib_2026-08-17.average_hash`) |
| `anchor_sig_path` | **主キー** | 対象フレーム±2フレームの縮小シグネチャ (`.npz`、`build_neighbor_signature`) への相対パス |
| `video` | 補助キー | 動画ID (`video_cXX` 形式)。**再DLで無効化され得るため参考値** |
| `frame_idx` | 補助キー | フレーム番号。**同上、参考値** |
| `t_sec` | 補助キー | 収集時の動画内時刻 (秒)。**同上、参考値** |
| `base_config` | メタ情報 | 候補グリッドを生成した収集構成 (フラグ文字列 + 収集日) |
| `wrong_cells` | 正解 | user記入。誤っているセルのみ `r3c2=1,r5c0=0` 形式、全部正しければ `ok` |

**主キー2フィールドが実質的な正解の型**であり、`video`/`frame_idx`/`t_sec`
は「今回はこの位置だった」という**参考値**に過ぎない。将来動画が変わっても
主キーで対象フレームを再特定できることが本規約の目的。

## 新規ラベル作成時の必須手順

1. 対象フレームを標準解像度 (1920×1080) に正規化して読み込む
   (`_label_anchor_lib_2026-08-17.read_frame_at`)。
2. 盤面ROI (`DEFAULT_P1_REGION`/`DEFAULT_P2_REGION`, `src/image_reader.py`)
   を切り出し、`average_hash` で16進ハッシュを計算する (`anchor_hash`)。
3. 対象フレーム±2フレームの縮小シグネチャを `build_neighbor_signature` で
   作り、`.npz` サイドカーとして保存する (`anchor_sig_path`)。
4. 候補グリッド (認識結果) を**どの構成で収集したか**を `base_config` に
   明記する。この構成自体が測定対象の場合、後段の判定で
   「グリッドを見て安易にokと書かない、必ず左の実画面クロップで判断する」
   ことをラベルシートのREADMEに明記する (自己整合性バイアス対策、
   `project_yardstick_first_results_2026-07-31` の教訓)。
5. `video`/`frame_idx`/`t_sec` は補助キーとして記録するが、**コード上で
   これらを主キーとして直接突合に使うことを禁止する** (次項参照)。

## 禁止事項

- `frame_idx` の完全一致だけで正解グリッドと測定対象グリッドを突合する
  コードを新規に書かない (再DL後に無効化されることが実証済み、W8)。
  必ず「主キーで対象フレームをまず再特定し、そのフレームの近傍で突合する」
  経路を通す。
- `video_id` の同一性だけで「同じ内容の動画」と仮定しない
  (`feedback_redownload_content_drift_2026-08-14.md`)。

## 再アンカリング手順 (動画が再DLされた場合)

1. `anchor_sig_path` から中心フレーム (offset=0) の縮小画像を取得する。
2. 新しい動画の `t_sec` 近傍 ±`SEARCH_WINDOW_SEC` (既定10秒、
   `_reanchor_yardstick_labels_2026-08-14.SEARCH_WINDOW_SEC`) を
   NCC (`cv2.TM_CCOEFF_NORMED`) で走査し、最良一致フレームを新しい
   `t_sec`/`frame_idx` とする
   (`_label_anchor_lib_2026-08-17.reanchor_by_signature`)。
3. 最良スコアが `NCC_CONFIDENT_THRESHOLD` (既定0.85) 未満なら
   `unanchorable` として明示的に記録する (黙って落とさない、W8の教訓)。
4. `frame_idx` 完全一致でのフォールバックは**測定側の最終手段**としてのみ
   許可し、使用した場合は結果に **`fallback_used=True` の注記フラグ**を
   必ず立てる (2026-08-17発見: ±0.35秒最近傍フォールバックがおじゃま
   一括着弾の境界を跨ぎ34セル規模の偽陽性を生成した事例、
   `docs/KNOWN_WEAKNESSES.md` W8(c))。

## ディレクトリ構成規約

```
data/verify/board_labels_<name>_<date>/
  labels.tsv          # sheet, anchor_hash, anchor_sig_path, video, frame_idx,
                       # t_sec, base_config, wrong_cells
  sheets/<sheet>.png   # 実画面クロップ + 認識グリッドの左右結合PNG
  anchors/<sheet>.npz  # anchor_sig_path が指す周辺フレームシグネチャ
  README.md            # 記入手順 + 既知の注意点
```

## 既知の注意点 (自己整合性バイアス)

候補グリッドが測定対象そのものの構成で収集されている場合、ユーザーが
グリッドに引き寄せられて誤りを見逃すリスクがある
(`project_yardstick_first_results_2026-07-31`: CNN/HSVの自己無矛盾では
検出不能だった教訓と同型)。**判断根拠は必ず実画面クロップ (左側)** とし、
グリッド (右側) はあくまで下書き・比較用であることをラベルシートの
README に明記する。
