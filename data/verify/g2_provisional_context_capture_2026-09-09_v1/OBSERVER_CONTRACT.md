# 暫定context捕捉契約

既存資産：最新 `g2_current_scope_capture/observer.py` の別SM sink、`g2_generation_publication_split/current_connection.py` の実isolate境界、固定 `g2_hidden_probability_capture/observer.py` のboard/PB serializerを再利用する。srcをimportせず、既に実factoryが構築した型と実objectだけを使う。新infer/normalize/モデル呼出は追加しない。

APIは `install(stack, collector, history, state, *, expected_frames=None) -> Recorder`。最新scope addonのinstall後に呼び、`state['provisional_context_observer']` へ登録。`finish(state)`、`verify(output, expected_frames=None)`、`guards()`、`REQUIRED` を提供する。live既定はhistory全29052〜36298/stride2の3624 update。PB/SM/currentの旧窓は不変で、candidate_scope_selectedに窓内外を保存する。窓外の旧票欠測は正常HOLD、外側full値は全て取得する。CPU小窓は明示expected_framesを渡し、縮小scopeをlive成功とは呼ばない。

保存は `provisional_context.jsonl`、`PROVISIONAL_CONTEXT_STATUS.json`、`PROVISIONAL_CONTEXT_RECEIPT.json`。1 full updateに1行、schema_version=`provisional-current-context/v1`、行の計装異常はfailures配列。source_id/run_id/frame_idx/time_sec/capture_token/available_frameを保持。updateには実引数clockと返却clock、call_index/pipe_index、returned、実active/end-lock/post-lockと各fieldのobserved、例外を保存する。available_frameは正常な実返却frameに限り、未来/不一致を成功へ補正しない。

sidesは1P/2P。before_hold/after_hold/finalはstate(enum.name)/state_value、confirmed/cnn/inferred全grid、probability全分布、NEXT/DNEXT、board_provenance/board_none_reason。pb/sm/next/candidate_rowは同update中に生成された既存原票の完全copy。identityの固定キーは `isolate_return_is_final / before_probability_matches_pb / before_probability_unchanged / after_probability_unchanged / final_probability_matches_after`。最初は実SideResultのis比較、残りは元objectを保持した全分布の比較であり、SHAだけや盤面個数では比較しない。PB欠測時のNoneと同じNone同士の誤合格を区別する。before/after PBとfinal PBの実is比較も別に保存する。

generationは実tracker.generationのbefore/after snapshot。software reset/actionは物理gameではない。gameはこの実collector/update入口が公開しない場合observed=false/value=null、理由を明記する。任意のgameやcanonicalは作らない。既知のtracker側side/pipe/clock不一致、観測器エラーはsticky失敗。reset前後差は隠さずbinderへ渡す。

canonical/ledgerは開始登録時からNOT_CONNECTED固定。ledgerの両side6fieldはvalue=null/availability=UNKNOWN/reason=producer_not_connected。既存観測器の失敗をこのunknownに変換しない。これは機械的な未接続の記録であり、会計値0・known・integrity成功の宣言ではない。全権限false。

元update/isolateは一度、返値identityと元例外identityを維持。観測の内部失敗はerrorsへ保存し、正常ENGINE/COMPLETE前のfinishで必ず拒否する。正常なlocked/nonstable/SM未呼出/PB欠測は取得成功と資格不足を分けてhold_reasonsへ保存する。完全scope不足は取得失敗。外側update終了後に全分布をもう一度コピーし、先のsnapshotと比較する。

初期実装は実collector採録ループやcanonical producerを新設しない。game/canonicalの未接続は明記し、親binderが現在候補用の別暫定正本として検収する。CPUの人工reader/私有初期履歴は実動画認証ではない。既存runtime/factory/prepared fixtureを一度再利用して全update/旧trace/PB非干渉を別試験する。
