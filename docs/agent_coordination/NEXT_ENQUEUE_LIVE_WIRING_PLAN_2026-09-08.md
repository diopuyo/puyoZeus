# NEXT 受理履歴を実 update へ接続する最小案

2026-09-08 JST、B の読取り設計。製造・追加 CPU replay・GPU・元コード変更は実施していない。ここでいう実接続は固定 video38 の開発用 shadow であり、汎用 live／本番採用ではない。

## 結論と製造判断

推奨する接続位置は frozen `src/recognition_pipeline.py:5117`／`:5135` の通常 enqueue 分岐そのもの。会計専用の `accepted_next` でこの分岐を一度だけ通し、原本の global `_last_seen_next` は従来条件で別に更新する。Counter／FIFO／metadata を持つ CPU owner の snapshot は外部へ戻さず、原本の通常 settle と metadata consumer をそのまま一度通す。

これで私有 snapshot 同期より小さい接続にはなるが、**原本 body 一回実行は原子的適用を意味しない**。原本は FIFO append、landing_pending、last_consumed_color、履歴を順に書く。途中失敗で失敗 side の部分適用を残さない契約は、既存 API では閉じていない。新しい履歴を加えることでこの穴が消えるとは主張しない。

したがって、実入力捕捉／AST の対象・非対象検査／実 body 一回性の CPU 製造は具体化可能。実適用を止める一点は、**元 FIFO の object／alias を保った append と、landing_pending・last_consumed_color・accepted_history を一括確定する既存 commit API がないこと**。対象はこの同 side の新規 enqueue だけであり、Counter／first_move／全 update／両 side／下流例外の rollback へ拡大しない。fail-stop だけを原子性へ読み替える案は採用候補にしない。

## 固定した根拠と既存機能の再利用

- A 文書 `docs/agent_coordination/NEXT_GRACE_INTEGRATION_BOUNDARY_2026-09-08.md` を全文確認。SHA `af4f159abe70be0d735a29d6824313366a6af65e2eac5190d8b6f6a801e2115f`。
- NEXT CPU source `scripts/next_enqueue_freshness_shadow_v1.py` SHA `4b05a8254607bd58212344c78a712633d5b49427d42cee0d91c00366fe363ead`、test SHA `00fd704ef1d5d8227ed52261ba99a611ae7c99bfec282c67428ed661dcd6d4ab`。41 PASS は CPU 私有 bundle の範囲。
- 以下の frozen pipeline は `.runtime_snapshots/event_first30_observed_context_v5_2026-08-30/src/recognition_pipeline.py`、SHA `6e945d7584025ae9803e14c9ac079b1a468f483d0beac681a1f0e580cdd10e02`。
- `scripts/initial_placement_accounting_shadow_v1.py:183` の `_compile_lines`／`:211` の `FrozenAccountingBlocks.load` は原本 AST の固定範囲抽出に再利用できる。clear／enqueue／settle の私有 CPU 対照用であり、実 pipeline の transaction API ではない。
- `scripts/diagnose_video38_c6_pending_commit_shadow_v1.py:347` の `_build_transformed_step` は、一意 AST 照合、元 filename／行番号保持、compile、変換 receipt の既存手段。`:372` は process 内 patch と復元。対象は `_step_side` 用なので、関数をそのまま update へ流用するのではなく同じ限定方式を使う。
- `scripts/diagnose_video38_initial_pair_observation_v1.py:191`／`:206` は主 NEXT と slide の実呼出を一回だけ観測し、返却 object をそのまま返す。`:176` は例外時 `returned=None` を保存し再送出する。追加認識や detector.update の再呼出は不要。
- 同 `:231` の update wrapper は返却後の独立観測用。**返却後だけでは当 frame の enqueue 制限に間に合わない**。`scripts/diagnose_video38_accounting_history_v1.py:95`／`:104` も設置時の update を closure に保持するため、全 wrapper 設置後の class.update 差替えだけでは frozen 本体を置換できない。

既存 CPU owner の `process:264` は旧 run SHA・保存 state・固定時計へ束縛され、`:219` で settle まで私有実行する。実 update にそのまま呼び込まず、受理条件・反例の正本として保持する。旧 run SHA を新 run の実観測認証として名乗らせない。

## 案 A：実位置の通常 body 一回＋会計専用履歴（推奨形、原子性は未充足）

### 捕捉と挿入の順序

1. fresh frozen load 後、固定 SHA の **元 RecognitionPipeline.update** に一意 AST 変換を設置する。history／start／観測 wrapper が元 update を closure に取り込むより前に設置し、その後既存 instrument を順に呼ぶ。closure の書換えや `f_locals` 代入は使わない。終了は ExitStack で逆順復元する。
2. frozen `:5061` の実主 NEXT と `:5077`／`:5089` の実 slide 戻りを、同 pipeline instance・同 side・同 update invocation・frame/time に束縛して捕捉する。slide は既存 wrapper の実戻り経路が使える。serialized row の `actual_call=True` という入力だけから実呼出権を作らない。
3. `:5096`～`:5103` の旧 NEXT／DNEXT 受理、`:5109`／`:5110` の slide 値は変更しない。物理・SM・連鎖終了へ渡す `next_pair_*` も置換しない。
4. 広い認識例外 catch `:5104` の**外側**、`:5117` の直前で会計用 pair と事前検査を準備する。実成功 slide、有限 non-bool の `0 <= diff_score < threshold_used`、pulse false、色 1～5、同一実呼出なら会計用 pair を採用する。cooldown=0 は要求せず、既存 threshold を変えない。欠測・実例外・高 diff は会計入力なし。内部 adapter の契約破損をこの認識 catch に飲ませない。
5. side ごとの元 enqueue AST `:5117`～`:5134`／`:5135`～`:5148` を、その位置で一度だけ実行する。ただし **この AST 内だけ** pair の入力を会計用 pair、`_last_seen_next_*` の参照／更新を会計専用 accepted_next に束縛する。FIFO append、landing_pending、last_consumed_color の原本処理は複製して別の後段で走らせない。
6. global `_last_seen_next_*` の更新は原本同様「active、旧 next_pair 非 None、両色 1～5」の条件で行う。quiet を追加条件にしない。値と実行順は各 side の元 global 代入位置から次の `:5169`／`:5188` の終了判定へ到達する前までに保つ。会計拒否中も global は元の値へ進む。一般 `_last_seen_next` の一時差替えと finally 復元はしない。
7. 原本 `_step_side`、`:7386`～`:7409` の通常 settle、以後の推論・collector へ自然に進む。新接続は Counter／first_move へ書かず、owner の私有 settle も呼ばない。実 MENU／TSUMO／action は変更しない。

「原本 body を再利用」と「無変更の全 block」を混同しない。会計入力と履歴アクセスだけが AST 差分であり、原本 global 更新は明示分離となる。変換前後 AST の差分許可リスト、左右一件ずつの対象数、残る append／通常 settle 呼出数を保存する。未知版や既に別変換された update は再構成を推測せず拒否する。

### 履歴の寿命と非所有フィールド

会計用 accepted_next に必要なのは、pipeline instance・side・実 reset 世代・実時計・前回の受理 pair・一回性だけ。Counter／FIFO／landing_pending／last_consumed_color の複製を継続保有しない。例外後や frame 欠落後に古い accepted_next を自動再利用しない。

- inactive は frozen `:4500`～`:4516` の既存 clear を保持し、会計専用履歴も同じ実 inactive に同期して失効する。初手 owner の「未消費観測を inactive 越しに保持」仕様を混ぜない。
- `pipeline.reset:3943` は SM などを reset するが、その関数内に Counter／FIFO／last_seen の明示 clear はない。一般 reset があったことだけで新しい空会計を認証しない。追加履歴は失効させても元 FIFO／metadata を勝手に clear しない。未知開始点の既存非空 FIFO へ accepted_next を推測で接合せず、その epoch の介入を拒否する。
- 連鎖開始・終了用 global NEXT は frozen `:4713`／`:4741`、`:5169`／`:5188`、`:5219`／`:5236` で使用される。会計専用の遅れをこれらへ直接伝播しない。
- `last_consumed_color` は `:7492` の infer_placement へ渡り、`:7778`／`:7780` で原本が消す。`landing_pending` は `:8039` の P7、`:8050`／`:8057` の grace を経て `:8053`／`:8060` で原本が消す。消去後に owner から書戻す経路を作らない。
- quiet 受理の時刻へ enqueue が変われば metadata 発行時刻も変わる。既知紫は旧 32698→候補 32704。grace の開始・期限や P7 も同値とはしない。ここを隠して「Counter だけの変化」と呼ばない。C の grace-only 修復を同時に初回比較へ混ぜない。

### 原子性が閉じない箇所

frozen 1P `:5126` append→`:5129` pending→`:5133` color→`:5134` history、2P `:5143`→`:5145`→`:5147`→`:5148` は一つの transaction API ではない。追加 accepted_next を先に確定すれば原本 append 失敗時に履歴だけ進み、後で確定すれば append 成功後の失敗で FIFO だけ進む。呼出順を変えるだけでは解消しない。

事前に色・clock・side・型・alias を検査することは必要だが、それだけで全書込が非例外・一括可視になる保証を作らない。Counter を直接変更しない本案でも、FIFO と物理 metadata と受理履歴の組が部分適用になり得る。元処理だから例外試験を免除すること、GIL／builtin だから複数文を原子的とすることはしない。

CPU 私有 bundle の一回 swap、pipeline `__dict__` 丸ごと swap、旧値の後付 rollback、消費済み owner の再発行、別 side 成功の巻戻しを代替にしない。要求単位は失敗した同 side の enqueue であり、全 update 巻戻しへ拡大しない。enqueue commit 後に infer／collector／記録が失敗しても、その既に成功した enqueue を未適用へ戻す要求はない。異常 run として保存し、正常 COMPLETE を出さない。

## 案 B：enqueue 四要素だけの commit 所有境界（責務案、まだ製造不可）

原子性を閉じるために追加で必要なのは、accepted_next・FIFO append・landing_pending・last_consumed_color の未確定準備と一括確定を扱う所有境界だけ。Counter／first_move は含めず、初手 owner／歴史会計と統合しない。commit 後は原本の同じ FIFO／metadata を通常 settle／infer 後 clear／P7 後 clear が消費する。これは現在の私有 AccountingView を毎 frame 同期する案ではない。元 FIFO を置換して既存 alias を切るだけでもない。

現 pipeline は複数属性・deque・局所 alias を直接使用し、原本 `:7390` が pending alias を取り `:7399` で pop する。既存の snapshot API を足すだけでは FIFO alias を保つ一括適用にならない。新しい owner に consumer 全体を移設することはこの最小案の前提にしない。既存 transaction から四要素だけへそのまま接続できる一括適用 API は、今回の対象資産にはない。

この局所 commit と alias 保全を具体 API で閉じられる場合だけ製造・独立検収へ進む。代替として巨大な全 pipeline 所有再設計を開始しない。歴史会計／現在公開の分離や初手黄の加算は含めない。本書で新 helper／モデル／source を製造していない。

## 実接続前に固定する CPU 反例と正常対照

以下は今後の製造に対する受入条件。今回の read-only 作業で実施済みとはしない。既知 128 side replay を保存先だけ変えて再実行する必要はない。

1. **実呼出捕捉**：fresh frozen の実 update を通す CPU fixture で、主 NEXT／slide 各一回、同じ return object、例外伝播、caller filename／line の保存を確認。cooldown 中 false＋diff>threshold の既知 32698／32702 両 side は会計だけ拒否し、global／slide／SM 入力は旧値。欠測／None 例外／NaN／bool／別 invocation／別 side の証跡を合成して許可しない。
2. **正常 NEXT**：保存された 32704 紫紫、32754 緑紫を境界対照として使う。cooldown>0 でも静止なら受理する。元 FIFO 順序、実通常 settle 一回、first_move 既存値保持、実 getter→既存 collector drain 一回を確認。旧 SM 固定 CPU の終端紫3・緑1は比較目安であり、修正後の全 update でもその値になると先取りしない。
3. **metadata 消費寿命**：enqueue→原本 infer success/failure の color clear→次 frame、enqueue→P7/grace→pending clear→次 frameを実 consumer で通す。消去済み値が復活しない。confirmed が None で旧 consumer が未消費になる分岐も保持し、勝手に消す／消費済みにする規約を追加しない。
4. **一回性／原子性**：各 side の FIFO append 前後、pending/color/history の commit 境界で失敗を注入。契約違反は適用前拒否、失敗 side に部分適用なし。1P enqueue 成功後 2P enqueue 失敗では成功 1P を保持し失敗 2P は未適用、正常 COMPLETE なし。別対照として enqueue commit 後の記録失敗では、成功 enqueue を保持して異常 run を残す。同一観測の再入・重複 update は原本全体を二度走らせて救済しない。案 A だけでこの条件を閉じられなければ実接続 PASS にしない。
5. **epoch／inactive**：32656→32658→32666 の active 揺れ、実 reset、frame 欠落、時計逆行、別 pipeline、古いキャッシュを試す。clear は原本どおり、初手観測を FIFO に輸入しない。A→A は新たな着手を発行しない。真正の同色着手を識別する API は未完成のまま明示する。
6. **実下流の非自明な差**：正常 1P 着地、1P cold 34080–34380／landing 34700–34792、2P 終端 35704–36298 を対象化し、P7／grace／infer／Counter 制約／chain start-exit／公開 HOLD・欠測・collector の missing-added-changed を分ける。quiet 遅延による物理の間接波及は許容差と決めつけず検収する。初回会計介入前の prefix は一致を要求するが、介入後の全旧行 bit-exact を合格条件にしない。
7. **設置・復元・保存**：元 update を保持する closure の実到達、単一 AST 変換、既存 C6／grace の `_step_side` が上書きされないこと、全 descriptor 復元、開始／終了 SHA、異常時 COMPLETE なしを確認する。追加判定 hook の例外が frozen `:5104` の catch へ紛れ込まないことも実 caller で確認する。

## 親へ渡す判断点

案 A の実位置・入力分離・下流一回性は製造可能な具体性まで絞れた。追加観測の GPU 再採取は不要であり、既存 source／保存値を使う CPU 接続検収から始められる。ただし **同 side の enqueue 状態束をどう一括確定するか** が未承認・未実装であり、現 41 PASS はそこを埋めない。

次に製造するなら案 A の実入力捕捉／AST 接続が最小。ただし適用を有効化する前に、案 B で示した四要素の局所 commit と alias 保全を一つの API で閉じる必要がある。この一点を未解決のまま実更新 shadow を完成候補として走らせない。初手黄2、Counter 後加算／clamp、架空着地、歴史会計／公開分離は引き続き対象外。

## 追記：未完了 enqueue 内部の限定撤回は案 A へ組み込める

親の最終要求確認を受けた読取り再検討。**本追記を前記の製造停止判断より優先する。** 禁止されている過去会計の事後補填／消費後 rollback と、下流に未移管の同期 enqueue の失敗撤回を同一視しない。確認した指示・資産には、後者の厳密な限定 transaction をユーザーが禁止した根拠はない。前記「旧値の後付 rollback」を後者まで含める解釈は広すぎた。

### 固定経路で確認できる隔離境界

- frozen `scripts/collect_boards_lean.py:2208` のループが `:2222` で同期的に update を一回呼ぶ。`:2259`／`:2260` の getter と `:2267` の collector 会計は返却後。`scripts/diagnose_video38_accounting_history_v1.py:167` も同 collector の同期呼出である。
- pipeline の両 side enqueue `:5117`～`:5148` の後に `:5271`／`:5292` の `_step_side` が順に呼ばれる。FIFO の同 side pop は `:7399`、color 消去は `:7778`／`:7780`、pending 消去は `:8053`／`:8060`。途中 enqueue の内部にはこれらの consumer 呼出がない。
- pipeline は `:747` の通常 class であり、確認した固定定義に `__setattr__`／`__getattribute__` override や該当 field の property はない。FIFO は `:1990`／`:1991` の通常 deque、metadata は `:1999`／`:2007` 等の tuple／None。対象 pipeline／collector／history に同 side 会計を並列消費する Thread／executor／async 起動はない。数値ライブラリの worker が存在しないという意味ではない。
- history の `:91`／`:145` は line trace を使わない経路。通常 snapshot は `:103` の update 前と `:108` の返却後であり、enqueue 中途には採られない。この読取りだけで外部の任意 thread、デバッガ、signal handler、未登録 callback が観測できないと普遍的に証明したことにはならない。

以上から、**既知の単一 thread 呼出境界と登録済み consumer に対する all-or-nothing** なら、案 A の内部で未完了差分を撤回する方法 (a) は製造候補にできる。既存 alias の参照先を替えず、原本の下流 consumer も移設しない。全 state 束の新しい所有体系は前提にしない。

### 最小 transaction の責務

1. 書込前に exact builtin deque、`maxlen is None`、元 object identity、長さ、既存 prefix の各 pair identity、plain dict 属性と immutable builtin 値、同 side／instance／thread／clock／epoch、非再入を検査する。旧 metadata/history を強参照で保持し、候補 tuple と結果証跡に必要な構造は可能な限り先に準備する。任意 subclass、descriptor、可変 payload、未知の observer は適用前拒否。元 FIFO のコピーを外へ戻さない。
2. その side の原本 enqueue を元位置で一度だけ実行する。commit 区間に推論／detector／emit／任意 callback／待機を挟まない。事前準備済み tuple を利用する AST 差分が必要なら、元演算の意味と許可差分を明示して検収する。global NEXT の従来更新は会計 transaction の外に置き、その値を撤回対象にしない。
3. 成功時は FIFO の対象末尾一個と metadata/history がすべて予定値になった境界を commit とし、以降に診断を記録する。新しい会計履歴の一回性も同境界で確定する。初期 pair を履歴に置くだけで append がない分岐も区別する。
4. commit 前の例外時だけ、FIFO が**同じ object、旧 prefix 不変、長さが旧値または旧値＋1、増分時の末尾が当該 pair object**であることを確認する。後者なら末尾一個だけを pop する。metadata/history は現在値が旧値またはこの transaction の準備値である field に限定して旧参照へ戻す。実行済みフラグだけに頼らず、append 成功直後・フラグ更新前の例外も区別する。色値一致だけで過去の同色 pair を消さない。
5. identity／prefix／長さ／field 値が予定外なら他の変化を推測して復元せず、recovery 不成立として実状態を保存し異常終了する。成功した別 side、Counter、first_move、原本 settle、commit 後の infer／collector／公開は一切撤回しない。撤回が成立しても実 update 全体の自動再実行はしない。

ここでいう不可視は「commit または撤回が終わるまで、既知の downstream consumer が読まない」という限定であり、CPU が中間値を書かないことでも、任意外部 observer に線形化可能であることでもない。追加の lock だけでは、その lock を使わない既存 alias 読取 thread を隔離できない。別 thread 観測／再入が必要な実行形では (b)、すなわち本方式を適用不可とする。

### 例外・検収の保証範囲

事前検査失敗、原本 append／metadata/history 書込中の通常例外、commit 前の一回の fault 注入について、対象 side の FIFO identity／prefix／内容と field 参照の完全復帰を実接続 CPU で検査する。型の偽装、bounded deque の先頭脱落、同色連続 FIFO、append 前後、再入、他 side 成功、commit 後の記録失敗を必須反例にする。fault 注入は専用テストの一回限りで、production API に任意 callback を公開しない。

MemoryError が書込前なら無変更、書込中なら同じ限定撤回を試みるが、**メモリ枯渇下の撤回処理自体や再度の非同期中断まで成功保証しない**。その場合は recovery_failed／COMPLETE なしを残し、状態復帰 PASS と偽称しない。KeyboardInterrupt／プロセス強制停止、外部 thread／trace による中途観測まで拡げた保証はない。builtin／GIL の存在だけでこれらを保証しない。

修正後の判断は、**案 A＋未完了 enqueue の限定撤回を CPU 製造・独立検収へ進めることは可能**。実装を止める条件は巨大な所有再設計の未完成ではなく、この固定呼出に対する「consumer 未到達・同一 thread・exact 型・当該増分 identity」の境界を実 CPU 接続で確認できない場合である。製造の実行許可は親判断待ちであり、本追記でコードや GPU は変更していない。

親は追記 (a) を CPU 製造対象として承認した。対象は新 `scripts/next_enqueue_live_shadow_v1.py`／`tests/test_next_enqueue_live_shadow_v1.py` のみ。既存 pipeline 行の非消費計装を理由なく排除せず、`sys.gettrace() is not None` の一律拒否はしない。固定計装に consumer／任意 callback がない scope を検査し、未登録 observer への一般不可視保証はしない。実同 side の復旧後再試行は CPU で一回性を検査するが、例外後の動画 update 全体を自動再走する機構は作らない。GPU は未承認。
