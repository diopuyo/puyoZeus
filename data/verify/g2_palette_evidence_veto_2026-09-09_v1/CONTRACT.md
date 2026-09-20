# 同call palette veto候補

default OFFの私有install。最終atomic journal後にstatic validatorだけをwrapする。原関数を元引数で一度実行し、原生成caller二code・実pipe/SM/context・同時計・board/frame/region identityを照合。同call localの前処理済みCNNを再推論せず使う。未知callerやscope不一致は失敗させ、silent素通しの成功にしない。原例外はそのまま伝播、閉鎖時descriptorを復元しframe参照を保持しない。

変更候補は可視row1..12で元色/原返値が両方1..5のセルのみ。同call CNN=元色、既存HSV彩度60以上・元色距離60以下、全5色距離で元色が唯一最小の時だけ元色を留保。距離比1.5はstd/revのペア比較に由来し、cell5色比較の根拠がないため採用しない。実黄2例も比約1.338であり、比1.5なら元の誤置換が残る。これは高信頼物理色の認証でなく、画像とCNNの双方が反対する履歴強制置換への局所veto。

原返値の占有mask・EMPTY/UNKNOWN/おじゃま・row0を変更しない。したがって原gravityが除去したセルは復元されない。再gravity/分類/resolveは呼ばない。画像欠測・低彩度・距離遠い・同距離・CNN不一致では原返値維持、理由を逐次JSONLへ保存。測定例外も記録し原返値を維持するが、品質検収で件数を確認する。元board/原返値/CNNをin-place変更しない。

候補の弱点はCNNとHSVが同じ誤色を支持した時に、本当の第5色誤読除去能力を失うこと。全色palette既知とは呼ばず、この反例をテストで明示する。旧API/defaultOFF、暗色/低彩度/同距離/元色敗北/欠測/原例外/入力非破壊/重力/隠し段と、実32798だけでなく正常実stepを独立検収する。採用/学習/会計/G2品質を自動発行しない。

src/scripts/凍結snapshot/旧unit/実行中run/本番flagsは編集しない。現在の途中票の局所CPU再現と、全run成功/非干渉/独立検収は別。
