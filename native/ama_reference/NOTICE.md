# 帰属と検証範囲

このクレートは puyoai の盤面処理と、配信用 ama_reference の
queue・評価・探索・道筋返却を Rust へ移植するものです。
ライセンスは MIT。元の puyoai の著作権表示と許諾文は LICENSE に保持します。
ama の原典の表示と許諾文も licenses/ama-MIT.txt に保持します。

基準は ef723d49ea31cc1fafbc7a31c5a2e2cd75b24725d023c2a9350ac99ab5913132 の exe。
同点の順序は、その exe の libstdc++ std::sort の比較・移動順に合わせています。
旧方式 F の API は変更していません。puyo_core に ama_reference_json_py を追加し、
Python の既存キャッシュ・色変換を経由して配信経路に接続します。
既定は OFF です。PUYO_OBS_AMA_RUST=1 のときだけ有効になり、
PUYO_OBS_AMA_RUST_THREADS でスレッド数を指定します（有効時の既定は 2）。
Python の GIL は検索中に解放します。検索の各枝は独立に実行し、元の枝順で集約します。
照合状況と速度測定の保留条件はジョブの RESULT_port2.md に記録します。
この変更は既存配布物への導入や公開を行うものではありません。
