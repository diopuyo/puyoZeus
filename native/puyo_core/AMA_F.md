# ama 型方式F API（明示呼出専用）

既存の `beam_search_py`、B3、Pythonブリッジ、採用フラグ、配信経路は変更していない。
`puyo_core` をビルドして以下の関数を明示的に呼んだ場合だけ実行される。
両者STABLEの `confirmed_board` を入力すること。認識器の出力を確定する処理は本APIの外側。

## 移植元と互換範囲

ama `dea210bcd92965ae08fbc311f23565b0fab6dbbb` の
`ai/search/beam/{eval,quiet,form,beam,layer,table}.{cpp,h}`、`node.h`。
MIT Copyright (c) 2023 citrus610。全文を `licenses/ama-MIT.txt` に同梱。
ハッシュは同梱rapidhashの80バイト入力版（BSD-2-Clause）、ヒープ順序は
計測exeのZig同梱libc++（Apache-2.0 WITH LLVM-exception）に合わせた。各全文も同梱。

Weight全15項目・build/ac・quiet鍵ぷよ最大3個・GTR/FRON/SGTR・置換表を移植した。
13行×6列の盤面なので `waste_14(0)=0`。14段目の盤面状態は持たない。
合法手は計測用C++と同じ自由22配置（同色なら11配置）。到達可能性判定は含まない。
補完はbag 0/3の2列、各手で見える本物ツモは手持ち/NEXT/NEXT2まで。
補完列は本物テープの4組目以降を参照せず、探索深さちょうどで切る。
連鎖計算は既存Rustの `simulate_chain(..., true)` を使う。

原典の置換表の16ビットaction、飽和eval、世代違いのhit、初期層のヒープ化を省く
実際の並び順を再現している。初期層の処理はC++標準上の前提を満たさないため、
他のC++標準ライブラリとの一致は保証しない。ここでは既存Windows exeをオラクルにした。
幅4/8/16・深さ6を全条件照合。幅の入力上限は20000だが、24要素以上の不完全層での
`std::sort` の同点順序まで一般化した互換保証はない。
置換表2枚は `Vec` が所有し、探索終了時に解放する。

## 単体F

```python
out = puyo_core.ama_f_py(
    grid, pairs, k=12, width=8, depth=6, colors=[1, 2, 3, 4],
    policy="main", incoming=60, target_score=0,
    profile="build", form=True, stop=False,
)
```

- `grid`: 行優先78要素、0/1..5/9。UNKNOWNは拒否。
- `pairs`: `(軸色, 子色)` のtupleの列。本物ツモを呼出側で固定する。4色集合外は拒否。
- `k`: 0..128、`width`: 1..20000、`depth`: 2..128。
- `policy="main"`: 候補の補完2列の最大連鎖得点の和で選ぶ。
  正の `target_score` はamaの探索打切り得点を変更する。0なら原典の95000。
- `policy="counter"`: 必要量で候補得点を頭打ちにし、見えている先頭2手の範囲で
  必要量を撃てる最短手を優先する。`target_score=0` は `incoming*70`。
  3手以降の最短性を証明する全探索ではない。
- `stop=False`: 計測オラクルと同じK手継続。`max_score` は最大の単発連鎖得点。
- `stop=True`（省略時）: 最初の実発火で停止。目標未達の小連鎖も無視しない。

出力はdict。`fired`、`target_reached`、`fire_hand`（1始まり・未発火はNone）、
`score`（最後の実発火）、`max_score`、`total_score`、`cancelled`、`excess`、
`fallen`、`pending`、`overflow`、`dead`、`coverage`、`post_fire_board`、`final_board`。
`steps` は `(手数, 軸列0..5, 回転0=上/1=右/2=下/3=左, 得点, 連鎖数, 消去後盤面)`。
本物ツモ不足は `coverage=False`。尾を乱数や補完列で実遷移させない。

換算はマージンなしの整数70点/個。単体の初期得点端数は0。
再帰は両者別に `score_remainder` を持ち越し、
`cancelled + excess = (total_score + carry_in) // 70` とする。
予告の未相殺分は終了時に一度着地させ、1回30個上限、残りは `pending` に残す。
`fallen` は満杯で入らなかった `overflow` を含む。単体APIの端数落下列は左から。
`post_fire_board` は最後の発火直後、`final_board` は着地も反映した盤面。
K手継続モードは能力計測用で、手の途中に来る別の攻撃や着弾時刻のモデルを含まない。

## 再帰候補木

```python
tree = puyo_core.ama_exchange_py(
    own, opponent, common_pairs, indices=(own_next_index, opponent_next_index),
    k=12, width=8, depth=6, colors=colors,
    levels=3, hand_ms=hand_ms, chain_ms=chain_ms,
    main_targets=[4200, 12600], deadlines=[6, 12], incoming=60,
    both=True, parallel=True,
)
```

`levels` は2/3。`chain_ms` は0〜19連鎖の20要素、0連鎖=0。
手時間・連鎖時間は外から渡す物理モデルで、学習の重みではない。
上の目標と期限は呼出例であり、採用済みの戦略閾値ではない。

`main_targets × deadlines` の本線候補に、予告全量/半量（個数切上げ）の対応候補を加える。
各候補で最初の実発火までFを走らせ、消去後の盤面・独立した両者のテープ位置を引き継ぐ。
相手の開始位置には先行して引いている組数を含める。自分の手数を相手の位置へ加算しない。
相手はこの連鎖時間内でFを走らせる。3段目は自分の連鎖が終わるまで待つため、
相手の連鎖時間をそのまま丸ごと自分の猶予にすることはない。

**盤面の時点の契約**: `opponent` とそのindexは最初の候補発火時点に揃えた盤面。
このAPIは「撃った後に返せるか」の条件付き評価を行う。最初の自分の準備中に相手が
どう置いたかの同時進行推定は行わない。現時点の両盤面だけから、その準備期間も含む
完全な同時対戦を再現するAPIではない。発火時点が違う候補には対応する相手盤面を
用意して別呼出するか、同一相手盤面を固定した条件付き比較であると扱う。
結果にも `board_basis="opponent_at_root_fire"` を必ず出す。

`own` / `opponent` に候補木を返す。`both=True` は向きを入れ替えた独立評価も実行し、
`parallel=True` はGIL解放下で `rayon::join` により両評価を並列化する。
各木のside=0はその評価の自分側。両向きは同じincomingを受ける別の仮定であり、
同時に二重送信される一つの対戦結果ではない。

葉の `balance = 相手の着地+残予告 − 自分の着地−残予告`（正が自分有利）。
`fired` と `target_reached` は別。`balance=None` は本物ツモ不足などで評価不能。
深さ上限で未読の反撃が残ると `truncated=True`。その収支は探索範囲末端の値であり、
勝敗確定やその後の全反撃を含む収支の断定ではない。親は子の数値を重みで混合しない。
各ノードに使用手数・index・開始/発火/終了msを残し、条件を追跡できる。

全消しボーナス、時間経過によるマージン、ランダム落下位置、着弾演出の所要時間、
観測補正によるchain_remainingは未統合。落下順序は再帰APIの `garbage_order` で指定可能。
候補を勝率へ変換したり、旧S式（残り色ぷよ−被弾）で選ぶ処理は持たない。

## 再現

`build_ama_windows.sh` はWSLの既設cargo/ZigとCPython3.12のimport libraryを利用し、
`target/ama/puyo_core.pyd` にだけ出力する。環境固有パスは `AMA_*` 環境変数で変更できる。
`scripts/run_ama_tests.sh` はLinux版を作業専用位置に出力し、venvへインストールせずに検証する。
`scripts/verify_ama_rust.py` は既存 `ama_bench2.exe` を変更せず各手/得点/手数を照合する。
`tests/fixtures/ama/` に入力・オラクルのSHA256・固定正解を保存する。
`scripts/run_ama_validation.sh` はテスト、照合、通常/低スペック計測を逐次実行する。

## 追補2: 版の差替えと既定OFFの再帰オプション

既定の重みは `ama_weights.json` の原文をビルド時に同梱して読む。
診断CLIはJSON/SHA依存を解決するため `cargo build --release --features ama-cli --bin ama_cli`
で作る（以前のrustc単体ビルドに代わる）。通常の拡張ビルドではこのCLIは作らない。
実行時のカレントディレクトリに依存しない。`weights_path` を渡せば指定ファイルを読む。
形式はamaのconfig.jsonと同じ `{ "build": { ... }, "ac": { ... } }`。
15重みを読み、原典で省略されている `form` は0、旧互換の `u:0` は許容する。
他プロファイルは許容するが、build/acの未知のキー・非整数・欠落・±10000を超す値は拒否する。
未知の機能を黙って無視して新版に一致したと扱わない。

`ama_f_py` / `ama_exchange_py` は戻り値の `provenance` に以下を追加する。

- `ama_commit`: 呼出側が指定した版。省略時は `dea210bcd92965ae08fbc311f23565b0fab6dbbb`。
- `implementation_commit`: 実際に移植した元ソースの版。引数では変わらない。
- `weights_sha256`: 読み込んだJSONの原文バイトのSHA256。空白・改行も印に含む。

既存のリストAPI `ama_search_py` は `with_metadata=True` で
`{candidates: [...], provenance: {...}}` を返す。省略時は従来のリストを維持する。
`ama_version_py(weights_path=None, ama_commit=None)` でも印を取得できる。
commit引数は版の記録であり、新版のC++コード自体を自動移植する機能ではない。
新しい版のロジック変更は照合の不一致を調査して移植する。

Windowsの作業用Pythonで次のように実行する。オラクルは保存済み足場と同じ
WEIGHTS / CFG / SEARCH / PLACEMENTS / AGENT / RESETプロトコルが必要。
新しいconfig.jsonはC++とRustの両方へ同じファイルを渡す。

```powershell
$py = 'C:/PuyoLive_20261014u/PuyoLive/python/python.exe'
& $py scripts/verify_ama_rust.py --oracle C:/new_ama/oracle.exe `
  --weights C:/new_ama/config.json --ama-commit NEW_COMMIT `
  --output native/puyo_core/target/ama/oracle_new.json
& $py scripts/verify_ama_rust.py --oracle C:/new_ama/oracle.exe `
  --weights C:/new_ama/config.json --ama-commit NEW_COMMIT `
  --boards tests/fixtures/ama/real_scenes36.json `
  --output native/puyo_core/target/ama/oracle_real_new.json
& $py scripts/bench_ama_rust.py --suite options --condition free `
  --oracle C:/new_ama/oracle.exe --weights C:/new_ama/config.json `
  --ama-commit NEW_COMMIT --output native/puyo_core/target/ama/options_new_free.json
& $py scripts/bench_ama_rust.py --suite options --condition lowspec `
  --oracle C:/new_ama/oracle.exe --weights C:/new_ama/config.json `
  --ama-commit NEW_COMMIT --output native/puyo_core/target/ama/options_new_lowspec.json
```

引数を省略すると同梱JSON・元の版・既設オラクルを使う。
`--suite original`（既定）は従来の幅/K/F/再帰計測、`--suite options` は追補2の比較。
`--stride 3 --repeats 3` は36盤面から12盤面を選び各3回。
`--configs full a b2 c d2 ad2` のように構成を限定できる。
受領票 `provenance` は上記3項目に加えてexe・拡張バイナリ・盤面入力のSHA256を保存する。
新しい検証結果は既存のgoldenを上書きせず別ファイルへ出す。

再帰APIに追加した任意引数は全て省略時OFF。

- `budget_ms`: 両向き合計で共有する壁時計の予算。0は即時打切り、Noneは無制限。
  浅い段を先に読み、同じ段では前段の余剰量、全量対応、目標得点/期限の順で優先する。
  優先度は安価な代理値であり、学習済みの価値推定ではない。
  探索の配置展開・各手・各ノードで期限を確認する協調打切り。
  途中の探索結果は手の決定に採用せず、完了した手だけを残す。
- `candidate_limit=2/3`: 本線の目標×期限を固定順で等間隔に2/3案へ減らし、
  対応は全量だけにする。標準の目標4200/12600・期限6/12なら、2案は(4200,6)/(12600,12)、
  3案はさらに(12600,6)。猶予で同じ候補になる場合は重複除去後に選ぶ。
- `level_widths=[8,4,4]`: 段別幅。配列より深い段は末尾の幅。未指定は従来のwidth。
  決定キャッシュは幅・深さもキーに含む。
- `both=True, parallel=True, num_threads=2/4`: 既存の両向き並列APIに専用プールの幅を指定する。
  独立タスクは2本なので、4スレッド指定でも探索を4分割する機能ではない。
  本番の共有プールを変更しない。プール作成・辞書生成も外側の実測時間に含める。

予算に達したFは `outcome.truncated=True`、その枝と祖先は `truncated=True`。
未読Fの `balance=None`。`coverage` は本物ツモ不足専用で、時間切れとは区別する。
未読候補を木から落とさない。予算モードの既読親には子が未読の際の暫定収支を残す。
`budget_exhausted` は時間切れの有無（既定OFFではキー追加なし）。
両向き逐次では先の向きが予算を使い切ると、後の向きは全候補が未読になる。
並列時は両向きが同じ絶対期限まで読み、片側ごとに予算を二重加算しない。
従来の深さ上限によるtruncatedも引き続き使うため、単なるtruncatedを時間切れと数えない。
協調打切りはOSの停止やスケジューラ遅延まで抑制するハードリアルタイム保証ではない。

比較専用の選択規則は、葉の生収支を自分が最大化・相手が最小化するminimax。
同点は本線→対応、期限昇順、目標昇順。未読Fを除き、既読の子がない親は暫定収支を使う。
一致率は時間内に候補が選べなかった場合も不一致に数える。葉収支差の母数は別記する。
APIの本番選択規則・勝率への変換を新たに配線したものではない。
比較元の「完全」は幅4・K12・3段の全候補木であり、無限深さのゲーム理論的な正解ではない。

## 追補4: 相手の準備とquick_counter（すべて明示指定時のみ）

`ama_exchange_py(..., advance_opponent=True)` は両盤面を同じ現在時刻のSTABLE盤面として受ける。
`indices` はそれぞれが次に引く共通テープ位置であり、先行分のずれを含める。
既定Falseでは従来の発火時点スナップショット契約と戻り値を保つ。

- `preparation_mode="shared"`: 同じ相手方策・固定見通しのFを1本生成し、各根候補の発火時刻の
  スナップショットを取り出す。`opponent_horizon=None` は根のK、`opponent_target=0` は通常のF本線。
- `preparation_mode="separate"`: 同じ見通しのFを候補ごとに計算する。共有との値の一致・実費比較用。
- `preparation_mode="candidate"`: 候補の発火までの手数を相手の見通しにする。同じ見通しは共有し、異なる見通しは別計算する。
  この方式では`opponent_horizon`を使わない。
- 根の`preparation`は方式、見通し、再利用、計算ms、開始/終了index、盤面、発火/終了ms、得点・送信個数・端数を返す。
  同時開始の根の準備だけを追加し、2/3段目の連鎖終了待ちは従来の時間軸を使う。
- 相手が先に発火してまだ連鎖中なら、その送信を自分の予告に加えて70点/個・端数持越しで相殺する。
  相手の連鎖終了後からしか相手の次の手は開始しない。自分の候補の置き方は開始時に選んだ方策を維持する。
- 相手が自分の発火前に連鎖を終えた場合、途中の着弾・再発火の再シミュレーションは未対応。
  `unsupported="opponent_landed_before_root_fire"`, `truncated=True`, `balance=None`を返し、枝を消さない。
  自分がK手内に発火せず相手の送信がその窓の後に到着する形も`opponent_attack_outlives_no_fire_horizon`とする。
  見通し不足、ツモ不足、準備の予算切れも理由を返す。これらを完全評価済みの枝として順位付けしない。
  `preparation_horizon`は、死亡などで必要時刻まで手を進められない場合も含む。元盤面・到達indexを併読する。

`ama_f_py(..., policy="quick_counter", opponent_consumption=0.3)` は催促への速い経路。
消費率が0.6未満なら、手持ち+NEXTの全合法配置を全列挙する。途中の最初の小連鎖で止め、
その後の連鎖時間を飛ばして2手目を置く経路は含めない。
`quick_visible=3`なら手持ち+NEXT+NEXT2まで列挙する。既定は2。
`quick_min_return=9`は相殺個数+送り返し個数の閾値。個数は既存時間軸と同じ整数70点換算で、真偽値へ丸めない。
成立する最短手数、同手数なら最小得点、同点は合法配置の列挙順で選ぶ。
不成立なら与えられたK・幅・目標で本線Fを最初の実発火まで走らせる。

`selected_policy`, `quick_complete`, `quick_max_return`, `quick_expanded`, 可視手数・閾値を追加する。
不足した本物ツモは補わず、全列挙未完了を「不成立の証明」とはしない。
`opponent_consumption=None`（省略時）は`assumptions`に催促扱いと本線扱いを両方返し、勝手に片方を選ばない。
`stop=False`はquick_counterでは使わず、常に最初の実発火までの方策とする。

比較専用の`ama_exchange_py(..., quick_candidate=True)`は可視全列挙の根候補を1本追加する。
可視手数・閾値は同名引数で指定可能。成立しない候補も`unavailable=True`, `balance=None`として保持する。
再帰の子は従来の本線/全量/半量候補。追加候補は実戦の既定へ採用しない。

旧bb2ed638バイナリで保存した90ケースの全戻り値を、追加provenance以外の
正規JSONバイトのSHA256で比較する。保存元バイナリSHAもfixtureに保持する。
既定オプションでの手・得点だけでなく、木・盤面・時刻・収支・キーの欠落も検出する。
