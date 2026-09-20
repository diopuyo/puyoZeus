# 私有 mutation journal 観測契約

`install(stack, collector, history, state)` を既存 NEXT / pending / grace / 浮き guard / 最新 T2 / W 設置後に呼ぶ。cold import は stdlib のみ。`guards()` を prepare の原 guard 集合へ追加し、閉鎖後 `finish(state)` → `verify(output)`。`REQUIRED` は atomic_journal.jsonl、ATOMIC_JOURNAL_STATUS.json、ATOMIC_JOURNAL_RECEIPT.json。

既存 generated `_step_side` の二つの固定 code identity / source SHA / bytecode SHA を検査する。再compile せず line trace で FIFO 消費、補正後 resolve 入出力、origin 開始、NEXT 履歴検証、persistence、glow、answercheck、最終公開前の実 locals をコピーする。原 infer / classify / resolve / writer の追加呼出はない。外側・内側の既存 trace を委譲し、元 profile には介入しない。終了時 trace / profile / wrapper を復元する。

resolve は実代入対象で二つを区別する。`final_board, chain_count` は着地本線 `resolve`、`final_b, _` は短い落下を補完する `resolve_secondary`。後者は発生した時のみ捕捉し、本線 34702 の票と同一視しない。全 run の未到達拒否は本線と共通後段に適用する。stage は原呼出直前 / 原呼出後の最初の line であり、origin 開始直後には active 属性がまだ代入されていない場合があるため、最終返却時の active origin と generation_after も保存する。

NEXT controller の既存 enqueue を外側から一回呼び、原 deque object / 元各要素の位置と identity / 末尾追加数 0–1 を検査する。観測 append に source / run / software reset / side / enqueue-call / slot token を付け、実 popleft の一個消費へ結ぶ。pair 色同値だけでは追跡しない。開始前 entry は UNKNOWN、software reset 後は旧 token と失効理由を残して UNKNOWN とする。これは物理 game / occurrence 認証ではない。

原 begin も一回委譲し、update 入口 Counter / FIFO と enqueue 入口を別保存する。inactive 中の原 FIFO / Counter clear は、実 begin の元 queue / 全要素 identity が一致し、enqueue 入力 active=False、当該 Counter と FIFO が空の時だけ、消費と別の `inactive_clear_between_begin_and_enqueue_writer_not_traced` として旧 token を失効させる。clear 行そのものは直接 trace しておらず、物理 reset とは呼ばない。条件外の unexplained mutation は拒否する。

全 3624 update × 両 side の step coverage と、全 run の主要 stage 到達を別検査する。MENU 等の原早期 return は events 空を許す。全 stage ゼロの完了は拒否する。CPU 小窓は expected_scopes を明示し、通常後段 stage の到達は必須。原例外発生時は保存失敗を recorder.errors に残し、原例外 object を優先する。認識成功でも観測異常は正常 receipt を発行しない。

全格子 / Counter / FIFO は detached JSON、画像はコピー・再分類しない。一 step の有限局所票のみ RAM に保持し JSONL へ逐次保存する。source / run / frame / time / side / software generation / pipeline object と実 code を同 token に束縛する。scope 前基点、物理 game、過去債務、integer 会計、atomic handoff、品質 / 本番許可は発行しない。

既存資産: snapshot recognition_pipeline.py の固定 AST 行位置、next_enqueue_live_shadow_v1.NextEnqueueController / EnqueueTransaction の原呼出、W.step_functions、最新 T2 test_guard → probe → runtime fixture / guarded_hash_cache を再利用する。人工 initial context / Counter / classifier / reader の CPU を実動画 writer 証明とは呼ばない。実 32798 NEXT 色置換や 34702 origin の証明は今後の新 run 側の責務で、旧成果物へ追記しない。
