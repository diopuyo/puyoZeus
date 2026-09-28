# E28 場面診断（評価修正なし）

ユーザーのE28指示によりE27（87b0fca）を新しい採用候補として記録した。
E27の場面基準不合格と原検収値は変更していない。production_config.py・src以下は変更なし。
レビュー動画はE22の保存コマンドにE27採用フラグを追加して生成する。

## 結論

指定区間の最初の原映像フレーム2755.017秒では、2Pは3段目消去中。
点滅消去群を除いて手転記すると、1Pは13連鎖87,720点、2Pは隠し段25通りすべて13連鎖75,440点。
差は12,280点、送り量は1,253対1,077個で、既存相殺5個を差し引くと純受け171個。
これは人が連鎖を読んだ場合の火力不足の確認であり、追加ツモを含む回避不能死の探索証明ではない。
2755秒より前に人が判断できた可能性は否定せず、指定区間内で確認した最初の時点を報告する。

2Pの発火時予測は0点・0連鎖・完走盤面なしのまま、通常途中予測はUNKNOWNで不採用。
隠し段用に適格な2755.417/2756.717/2760.817秒も各1観測しかなく、必要な連続2観測に達しない。
診断だけで1枚から列挙すると、それぞれ25/5/5通りが次段得点にも全件一致し、最大75,440点になる。
本経路の採用は0件であり、この事後照合をE27の採用件数に加えていない。

2755〜2771秒は480観測、着弾再評価89回。完走未確定86回・盤面静穏待ち3回で止まり、複数着弾探索は0回。
2770.617秒に連鎖終了の静穏確認を通過しても、盤面/NEXTの静穏待ちが残る。
2770.717秒に交換が終了し、死亡保持がないため残り171個の台帳だけでは着弾評価を再開せず、通常G_feへ移る。
回避不能死は対象場面で一度も成立しない。通常評価の2P≤5%は生値2771.150秒、表示2771.250秒。
映像からの実死亡確定は2773.783秒（目視の火力不足確認から18.767秒後）。
目視確認との差16.233秒は、途中盤面の2観測不足→完走未確定→終了後静穏待ちと交換終了が原因。
単なる表示EMAの遅れではなく、最後の平滑化差は0.100秒だけである。

## 隠し段「次段一致2件」の場所

- 固定3動画の2件：fcの342.933秒・524.967秒、いずれも2P。今回の場面ではない。
- レビュー区間の別の2件：2682.717秒・2732.150秒、いずれも1P。今回の2Pではない。
- いずれも隠し段上限を使った死亡探索は0回。得点一致・入力への利用・死亡探索を区別する。

## CSVの読み方

SCENE.csvは指定区間の全480観測。reevaluated=Trueの89行が着弾評価を再実行した時点。
SCENE_CONTEXT.csvは2750〜2773秒も含む。時刻単位は原動画の絶対秒、得点は点、ojamaは個。
空欄は未計算または未取得。p2_predicted_scoreの0は初期予測値で、最大打ち返し0個という意味ではない。

- t_sec/game/p2_state/source/p2_probability：時刻、試合、2P物理状態、評価元、生の2P確率。
- landing_evaluation_sec/reevaluated：最後に着弾計算した時刻と今回の再実行有無。保持値を現時刻の計算と混同しない。
- p2_observed_chain_count/p2_formula_score：観測段数と当該連鎖の累積式得点。通知タイミングにより段数が1観測遅れることがある。
- p2_midchain/p2_hidden/p2_hidden_samples：通常途中予測・隠し段死亡専用予測の状態、連続落ち切り観測数。
- p2_hidden_unknown/p2_visible_unknown/p2_floating：隠し段・可視段のUNKNOWN数、浮遊セルの有無。盤面なしは空欄。
- p2_predicted_score/p2_predicted_chain_count/p2_completion_certain：保持されている完走予測と死亡判定に使える確実性。
- p2_hidden_maximum_score：実採用された隠し段候補の最大点。本場面はすべて空欄。
- p2_maximum_extra_counter_ojama/p2_near_future_ojama/p2_resolving_credit_ojama：追加応手の楽観上限・近未来火力・既発火相殺の加算分。最大/近未来探索は本場面では未実行。
- p1_formula_score/p1_provisional_send_ojama/p2_provisional_counter_ojama：観測を反映した送受け両側の暫定総量。上記の目視完走予測とは別。
- p2_net_incoming_probability_ojama/p2_net_incoming_death_ojama/p2_pending_now_ojama：確率入力の純受け・直近死亡計算の純受け・現時刻の独立台帳残量。
- p2_required_cancel_ojama/p2_hands：単発死亡探索の必要相殺量と手数。相殺未計算は空欄。
- p2_safety/p2_death/p2_multi_reason：追加整合条件、死亡成立、複数着弾を止めた最初の条件。
- p2_multi_nodes/p2_multi_rounds/p2_multi_detail：探索ノード、各着弾のincoming/maximum_send/cancelled/dropped/remaining/responses、原診断JSON。探索前に止まるためroundsはnull。
- p2_final_non_death_condition：当該時点で最終的に死亡へ進まない条件。交換終了後は過去の計算を現在の計算と扱わない。

SCENE_RAW.jsonは各観測の凍結コピーで、連鎖ID、台帳内訳、得点換算時間、全評価値を残す。
Gitには同内容のSCENE_RAW.json.gzを保存する。
SINGLE_SAMPLE_DIAGNOSTIC.jsonとSCENE_CONCLUSION.jsonは診断専用列挙と次段照合。
VISUAL_VERIFICATION.jsonは最初の原映像からの手転記・独立計算で、未来の最終得点を入力にしていない。
REPLAY_VERIFICATION.jsonでE27と表示・イベント・診断が完全一致することを確認。
live/SCENE.csvは納品動画自身の入力を再生した同じ形式のCSV。live/REPLAY_VERIFICATION.jsonで動画出力との完全一致を確認。
LIVE_COMPARISON.jsonは固定記録と納品動画で死亡台帳・採否条件・複数着弾診断が一致することの検査。
2755.017秒のパネルに出る489個は確率用受け量で、死亡専用台帳の490個と用途が違う。動画と固定記録の差ではない。

再生成：`python -m scripts.diagnose_e28_scene` → `python -m scripts.report_e28_scene`。
目視転記の検算：`python -m scripts.verify_e28_visual`。動画：`python -m scripts.render_e28_review`。

## 納品動画

`logs/review_zenchi_g41_43_e27/overlay.mp4` は359,943,511 bytes。
`D:/puyo_analyzer/videos/review/zenchi_g41-43_e27_mobile.mp4` は21,768,341 bytes（21.77 MB）。
198.267秒・30fps・5,948フレーム・横1280、両方AAC音声あり。
スマホ版はCRF28・映像上限1,000kbps・音声128kbps、全編デコードエラー0。
冒頭・途中・指定場面を目視確認済み。完成後、重複する無音中間動画だけ削除した。
元映像は既存共有資産のため残している。動画本体は指定パスに保存し、Gitには検証値・ハッシュ・再生成コードを保存する。
