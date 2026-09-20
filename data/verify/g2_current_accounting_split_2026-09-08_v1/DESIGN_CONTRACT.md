# 現在復帰と過去消去会計の分離：最小契約

2026-09-08 17:13 JST。案Bのユーザー承認後。私有ownerのCPU契約から開始し、本番・学習・実pipeline採用とは分離する。

## 目的と範囲

旧pendingの存在だけで正当な次手の現在盤面も公開不能になる契約衝突を解く。旧ChainCommitTransactionは同一actionのboard/Counter一体契約として一切変更しない。新契約は旧effectの部分適用ではない。

現在盤面はfresh観測とSTABLEおよび出所を検証する独立policyへ束縛。過去消去はimmutable originへ束縛した別債務とし、actionが進んでも同gameで清算できる。ただし新game/resetとの混同を拒否する。失効したprediction ledgerを復活・旧action偽装して追記しない。

## 保持する安全条件

- current更新は会計・過去証拠・消費IDを変更せず、late settlementはcurrent全grid/revisionを変更しない。
- board確定権とaccounting確定権は別policy。未接続・任意boolは拒否。人工policyでのCPU正例は物理的真値証明ではない。
- source/run/side/game/reset/action/観測frame/timeを照合する。未来に利用可能な証拠の遡及適用、時刻逆行、UNKNOWNや非STABLEを正常盤面として許可しない。
- Counterは同gameの正当な基点・着手増分・確定消去履歴に結ぶ。内容とrevisionを両方検査しABAを拒否。origin再登録や二重精算を拒否。
- 後続の正当な着手増分を消去しない。未対応の履歴・基点はunknownとして停止し、Counterのclampや後付補充で通さない。
- 未清算または基点不明ならaccounting availabilityはfalse。current復帰からT2/フィルタ/おじゃまledger/M1/学習を自動許可しない。
- 私有single-writer ownerの不変stateを交換する。変換・検証は交換前に完了し、policy callbackの再入を拒否。CPU合格から外部pipeline複数属性の原子性を推定しない。

## 独立検収

正例は人工入力で旧origin債務保留→新actionのfresh current復帰→正当な着手増分を保持したlate精算→current不変を検査。新action後の旧board上書き・債務が消えたことだけによる公開許可・未清算availability=true・二重精算・他game混入を反例化する。

製造の凍結後、親がsource/test全文と実保存結果を検収し、別processで境界検査する。旧API regression、例外時不変、policy再入、Counter revision ABA、将来証拠拒否を含める。

実証拠producer/availability consumerはまだ未接続。この限定CPU契約が通ってもG2 CLEARではない。次は既存再取得候補を真値扱いせずfresh current証拠生成と会計基点の実入出力を接続する。正常ケースが永続unknownのままならG2不合格。

## 担当と工程

Codex独立担当はsplit_contract.py/test_split_contract.pyと製造CPU記録、親は本書・Opus限定反証・親独立検収。最大Codex2、通常Opus1を必要な設計反証に限定、Fableなし。新GPUなし。物理映像の検収は接続CPU合格後の別工程で行う。
