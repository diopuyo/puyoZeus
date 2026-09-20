# 限定実動画・早期退出修復

背景は正常完走 `video38_full_observation_live_2026-09-09_v1`。3624時計、全域W/PB/SM/contextを採録。
介入は2P34620–34702（42時計）だけ。旧1P chigiri B窓34468–34516には介入しない。
元stepを一度呼び、内部で固定浮き一個guardを設置・復元する。
独立根因診断を受けた私有修復であり、src／snapshot／本番設定変更や保存SMの再注入はない。

観測専用FULL_OBSERVATION_SCOPE_RECEIPTの「全frames同一」は、意図的修復と両立しないためこの新契約では生成しない。
旧split runnerのraw/NEXT全域同一、firstC6以前全stream原文同一、全来歴/SHA/正常終了/排他保存/finally後ENGINEをそのまま保持。
実run差分は正常背景と別途集計し、1Pと修復前34682未満の非干渉、FIFO消費・infer回数・有限復帰を確認する。
過去の失敗runを成功化しない。新publisherやlanding_color_fixを同時接続しない。

局所ガードは34682/34686だけ拒否し、34702でSMが有限退出することを要求する。
それだけでは公開9色、T2旧色復元、精算在庫、隠し段確率較正、実2side M1入力やG2全体の品質PASSにはならない。
