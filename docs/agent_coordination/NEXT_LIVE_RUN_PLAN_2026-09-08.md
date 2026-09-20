# NEXT 単独 shadow の実動画比較計画

2026-09-08 JST。B の実行前計画のみ。新 runner／launcher／計測器は製造しておらず、GPU も起動していない。親の独立 CPU QA が実行依存であり、ユーザー回答待ちではない。

## 比較する一軸と固定候補

今回の一軸は「実 NEXT/slide を再呼出せず、会計専用の受理履歴で高 diff 中の誤 enqueue を抑える」こと。既存 start_epoch 修復を両群共通の背景条件として残し、NEXT だけを追加する。chain_end v2／post-chain grace／初手黄2適用／歴史会計・公開分離を追加しない。grace 単独 run の結果を対照に使わない。

候補は `scripts/next_enqueue_live_shadow_v1.py` SHA `c2de2e33f4970cf5dc4bf4ec6e8cece03ab40d821b5fd44acd4063e1a1339ffa`、test SHA `2a1d0163f841f8dfdb5857134bde995aefd2af8e1aeda24445e302ad0f78ce16`。47 PASS は人工 reader／認識結果、正常着地用の人工 detector 信号を用いた CPU 接続検証で、動画認識・物理真値・全時間帯の非劣化ではない。独立 QA で候補が更新される場合、本計画を旧 SHA のまま実行許可へ使わず、新固定版へ再結合する。

新規予定 root は `data/verify/video38_next_enqueue_live_shadow_2026-09-08_v1/`、ログは同名 `.log`。既存 root／ログがあれば排他拒否し、上書き・自動削除・別名の無断再走をしない。

## 必要な履歴と開始点

NEXT 履歴を保持する対象は **frame 32494～36298 inclusive、stride 2、1,903 update**。しかし動画を 32494 へ cold seek して始めない。元比較と同じ **484.2～605.0秒、frame 29052～36298 inclusive、3,624 update** を同じ pipeline instance で走らせる。32494 より前の 1,721 update も保持し、認識器・score／latch・SM の前史を変えない。

再利用する `scripts/diagnose_video38_accounting_history_v1.py:177` の run は途中で pipeline を作り直さず、`:188` の `start_sec=484.2`／`precise_seek=False`、`:189` の `sample_interval_sec=0`／`normalize_fps_30=True`、`:82` の絶対 frame 時計検査を持つ。32494 の reset は既存 start_epoch の限定適用をそのまま使い、新しい試合開始認証を作らない。

元動画は `data/frames/video_38.mp4`、SHA `b3728078cc2e8282065e5dd78ca1c13e6a443ffdf1dc4333e3da06f0852757d3`、既存確認値 1280×720／60fps。decoder が実フレームを読み 1920×1080 へ既存変換してから認識する条件を保持する。新動画・DL は不要。

## 対照と比較母集団を先に固定する

主対照は `data/verify/video38_start_epoch_shadow_2026-09-07_v1/`。COMPLETE SHA `0dee7d346a0231ed3f95347d549b1a1e5436dc512ce405e5f597db6587b0c326`、frames SHA `41651413c89ca722d1bd9e4db3a11a80c49d57b9d8a17f6ab69963506c0ad30c`。COMPLETE が列挙する全4 artifact を開始・終了 guard へ含める。

初手観測 v2 の root は補助証拠。COMPLETE SHA `c87ceadd5e07fd78284db101bb33c879165615e799cd991e8ef80b6bcef012e9`、frames SHA `d2d012972fb61092b1ecae1e49e04ed1bf3b454110028bb71ac75358df005b67`。`INDEPENDENT_RUN_QA.md` と `INITIAL_PAIR_OBSERVATION_SUMMARY.json` により、追加 `initial_pair_*` を除いた **15,722 行は主対照と caller／全文字列／順序まで一致**している。二つの対照を適宜使い分けて良い方だけ採用しない。

新 run では inactive の追加 CNN 推論を外す。黄2の可視性は既に保存済みであり、NEXT 会計修復は inactive 観測を pending に投入しないため。除外するのは初手観測の4種 `initial_pair_main_next`／`initial_pair_main_slide`／`initial_pair_result`／`initial_pair_inactive_next` の追加行であり、元 start/end gate 計装や既存 start_epoch 行は保持する。旧追加行数は main 159、slide 314、result 391、inactive 232、合計1,096。追加 inactive 推論の旧実測時間は0.618秒なので、これを大きな高速化とは主張しない。

主対照の15,722行の内訳を今回の読取りで確認した：frame_side 7,248、accounting_update 7,248、collector_snapshot 247、start_gate_latch/result 各391、start_gate_board_return 2、start_gate_reset 1、end_boundary_detector_return 68、step_enter/return 各44、exit_helper_return 34、boundary_repair_mutation/observation 各1、seek/target_decoded_frame 各1。

固定するのはこれらの **kind と key=(kind, frame, side, occurrence) の母集団**。候補の collector や状態依存 event 行が15,722へ揃うよう削除・補完しない。frame_side／accounting_update は全3,624×2の完全順序 coverage 必須、collector／条件依存行は全 missing-added-changed を保存する。

## 最小 runner 接続：既存処理を再利用し、意味の違う finish はそのまま使わない

1. `history.prepare/run/collect/finish` の動画・モデル・source・絶対 clock・保存遮断を再利用。既存 `diagnose_video38_boundary_repair_shadow_v1.py:84` の start_epoch prepare と依存 guard、`:110` の固定 module load を再利用し、背景 start_epoch を同じコード／証拠で設置する。
2. NEXT の `install(stack, collector, rec)` を **history の update closure 保存より前**に呼び、その後既存 start_epoch の instrument を呼ぶ。共通 `instrument:145` は「既存計装の後に修復」の順序なので、新 NEXT mode と称してそのまま呼ばない。既存 source を書換えず、新しい短い adapter が設置順だけを調整する。生成 update は元 globals を共有し、後続 helper patch と `_step_side` へ追従することを独立 CPU QA と結ぶ。
3. 初手観測 v2 の `instrument` 全体は呼ばない。必要な実 NEXT/slide 証跡は、NEXT controller が既に捕捉した同 invocation の **実 return** を読み、既存 `initial_pair_observation_v1.py:90` の `pairs` と `:96` の slide 値表現、history の `accounting_snapshot:35`／`rec.emit` を再利用して保存する。新しい detector、再推論、差分閾値、疑似 pulse は作らない。
4. 追加保存は当該 invocation の前後へ薄く接続し、transaction **内部**に callback／emit を挟まない。実 return の同一性、呼出数、raw NEXT、slide pulse/diff/threshold、global NEXT、会計7 field／accepted_history の前後を残す。終了時に失敗した保存を認識の広い catch へ飲ませない。既存 capture を再度呼んで証拠を生成せず、返却済み値の受渡しだけを行う。
5. `initial_pair_observation_v1.compare_legacy:296`／v2 `finish:168` の全15,722行完全一致は **観測専用**の契約なので NEXT 修復には使えない。共通 `changed_fields:217` の欠列≠None、`read_rows:157` の occurrence、`compare_rows:228` の全差分保存を再利用し、partition と前処置済み対照の扱いだけを adapter で変更する。
6. 共通比較の `:233` は既存 boundary_repair 行のある対照を拒否する。主対照にある start_epoch 2行は「既に適用した背景」で、NEXT の新 mutation ではない。両群の元行として保持し、NEXT prefix だけを別に分離する。新種の `next_enqueue_live_decision` を既存 `validate_repairs:174` がそのまま理解できると偽らない。

新 adapter に必要なのは設置順・既存 capture の保存・partition・最終 receipt の接着部分だけ。同じ recorder／collector／動画抽出／比較器を丸ごと複製しない。追加コードの製造は親承認後。

## 一回の実走で確かめること

- **介入前**：既知の最初の旧誤 enqueue は frame32698。これより前の全主対照行は全文字列／caller／順序一致を要求する。差分を見た後に prefix 終点を動かさない。内部 accepted_history の初期化と、元会計へ現れる最初の効果を分けて記録する。
- **実 NEXT 非変更**：全対象1,903 updateで主 NEXT/slide の実呼出有無・回数・例外・同 return を記録する。常に1回と補完せず、inactive／prev_frameなし／元例外による未呼出を分類する。535–548秒は旧初手 v2 の保存実 return と共通観測があるので、同 clock／side／occurrenceで直接比較する。以後の raw NEXT は旧側に同じ詳細が未保存であることを明記し、全区間 raw bit-exact と主張しない。global NEXT は全区間 accounting_update の既存列で比較できる。
- **既知誤色と正常手**：32698／32702 両 side の高 diff false を会計拒否し、正常紫32704／緑紫32754を失わないか。旧 SM 固定CPUの紫3・緑1／FIFO空は目安であり、実SMへ接続した後も同値になると先取りしない。first_move、metadata発行時刻、P7／grace／inferの意図差を併記する。
- **初手は未解決**：実544.0空→544.7黄2着地→546.367六色の保存 PNG／PHYSICAL_QA を再利用し、黄2を新 owner が後から足していないことを確認する。Counterの一部が改善しても accounting_basis_verified／G2 CLEAR をtrueにしない。
- **正常窓**：1P cold 34080–34380、1P landing 34700–34792、2P終端・次手35704–36298の公開HOLD／欠測／復帰時刻、collector missing-added-changedを固定して検収。両sideの全 frame_side を残し、片side改善だけで他sideを無視しない。元背景には既知の連鎖問題が残るため、NEXTだけで真の終了・次手完成とはしない。grace単独結果から正常回復を借用しない。
- **同色 NEXT**：変化のないA→Aから新 event を作らないことを確認する。値が同じ行が何件あっても、それが実際の同色着手だった証明にはならない。元動画／既存証拠で真正の同色着手を同定できない箇所は未認証とし、この一走で同色課題を完成扱いしない。
- **一回性**：実 getter／collector drain は通常 settle 後の元経路だけ。metadata消去後の再注入がないかを採録し、未知 before/after を success で補わない。復旧例外の故障注入は CPU のみで、GPU診断に任意callbackを持ち込まない。

## 完走／異常停止／実 device の最小検収

history `:119` の load receipt は `next(cnn._model.parameters()).device == cuda:0` と frozen src を確認する。既存 model load guard、実 source SHA、元動画 SHA、decoder clock、3,624 frame、必須14,496 side行を維持する。GPU利用率だけで完了・稼働を判断しない。

開始時 guard は元対照4 artifact＋初手 v2 の補助5 artifact／独立QA、既存 start_epoch source/test/証拠、NEXT source/test/GUARD_PATHS/REQUIRED_INPUT_SHA256、新 adapter/test/launcherを含める。Windows/WSL path解決と開始SHAの同値を検査し、保存後・最終COMPLETE直前も再hashする。

旧 `launch_video38_initial_pair_observation_v2.sh:14` と共通 launcher はdetach開始PIDだけで child の数値exit codeを保存しない。PID消滅をexit0と扱わない。既存 `launch_video38_post_chain_grace_shadow_v1.sh:19` の実child終了値取得、対応runner `finish:312` の ENGINE_COMPLETE保留、`finalize:346` の CHILD_EXIT＋全SHAを結ぶ手段を**終了保存の共通機構としてだけ**再利用する。graceのmodule／mode／root／盤面処理を importして起動しない。

recognition child exit0、独立finalize成功、全guard／artifact一致が揃って初めて診断COMPLETE。例外／recovery_failed／中断／数値exit不明／部分artifactは異常証跡として保持し、正常COMPLETEなし。commit後の記録失敗を理由に成功会計を過去へ戻さず、同じ動画を自動再走しない。COMPLETEは `experiment_complete_not_adopted` であり、品質合格ではない。

## 実行許可の直前チェック

親の独立47件＋実frozen接続QAが固定候補SHAへ結ばれた後、短い adapter の CPU 結合検収を行う。主要条件は closure前設置／後helper追従、元認識呼出増加0、既存source不変、前記partition、元背景修復行保持、全復元、実child失敗時COMPLETEなし。これが通れば親が新rootへGPU一本を起動する。Bは重複起動しない。

今回追加したのは本計画だけ。初手・歴史会計の未着手を先行実装せず、親QA待ちをユーザー作業待ちへ読み替えない。

## 実返却型P1の限定修正へ再結合（親承認後）

上記c2de2e33／2a1d0163は旧版の証跡として保持する。実`NextDetectionResult.next_pair`は固定immutable propertyだがアクセスごとに別tupleを生成するため、旧版の再読identity比較は実入力を拒否した。新runner用CPU試験で旧版の同値true／identityfalse→ValueErrorを一度実測してから修正した。

現候補source SHAは`e5ebff6827119c616319b4598fce1428c98643b621ff02e985736b43783d9237`、test SHAは`3aba1e71d7e915d5875d06489936917a5a0719ac881ed9d98d37fc0b85e1eb66`。主戻りobjectを同pipeline／invocation／clockに捕捉する条件は維持し、固定`NextDetectionBothResult`／両side`NextDetectionResult`／exact int列だけを許す。固定`SlideMotionResult`以外も拒否する。pair照合だけは、その捕捉済み同side immutable propertyとの値整合に変更する。任意objectや別side／別clockを色一致だけで許可しない。

更新版56 CPU PASS／11.82秒。従来の実update／原本settle／metadata消費／collector drainも実dataclass返却へ置換した上で通過。固定32494の空FIFO／score0相当CPU入力で、busy中の元update4868→元reset、両side epoch1・inactive・NEXT未呼出を確認した（OCRは人工入力、真の試合開始を自動認証する試験ではない）。旧47 PASSの独立QAやFable所見を更新版の独立PASSへ流用しない。親の更新版独立QAと新3fileの接続QAがGPU起動依存である。
