# 24h の単一勝率の試験設定（2026-10-28、既定OFF）

24g2を土台にした既定OFFの配線。既存の配布物・既存の旗の既定値は変更しない。
`live_24h=true` の起動だけ、旧v4.9f・ama別欄・線1の表示選択を止める。
数字と両グラフは同じ公開確率を使い、確定終局の規則は維持する。

既存のC2用設定に、次の項目を加える。開発CLIでは同名の `--live-24h`、
`--live-24h-ama` / `--no-live-24h-ama`、`--live-24h-delay-ms`、`--live-24h-model` を使える。
利用者ランチャーのJSONにも対応する。相対モデルパスは利用者JSONの場所から解決する。

`live_24h_retr`（CLI は `--live-24h-retr`）は既定false。trueかつama ONの時だけ、
EX計算子へ `PUYO_EX_RETR=1` を渡す。状態更新子はOFF。PC2の `7ae53bf7` の抽出を使い、
retRの8列もEXと同じ遅延・保持・解除・期限で配送する。モデルが宣言したretR列だけを推論へ渡す。
現行L200Hモデルの場所は変更しない。retRモデルの準備後は利用者JSONの `live_24h_model` を替えて再起動する。
新しい同梱pydの指定は組立台本だけに反映してあり、この変更で組み立ては行わない。

```json
{
  "live_24h": true,
  "live_24h_ama": true,
  "live_24h_delay_ms": 200,
  "live_24h_model": "models/ex/ensemble.json"
}
```

- 前提は `b3_judgment`、`b3_async`、`b3_m3`、`b3_c2`、`b3_skip_icon_reading` がtrueの既存C2構成。
- 主旗OFFなら従来の設定・出力を維持する。生成するpipeline設定に24hのキーは足さない。
- 主旗ON・ama OFFならC2だけを推論し、EXの観測・探索子・モデルは作らない。
- 主旗ON・ama ONならC2特徴の計算子は行だけを返し、親でEX40列を加えてC2+EX ref3だけを推論する。
  EXが全欠測でも同じモデルを使い、C2へ切り替えない。各refの平均方法は既存マニフェストの指定に従う。
- モデルは `b3_ensemble_v1` のマニフェスト。各memberのSHA256、3本、C2+EXの列と順序を検査する。
  監査30列・配信に存在しない特徴は拒否する。モデルを入れ替えるときは新しいパスを指定して再起動する。
- OBS試験入口は `NETNOICONEX_REGEN_L200H_final_ref` のs1003/s1004/s1005を使う。

ama ONでは起動環境に `PUYO_LIVE_EX_POLICY`、`PUYO_LIVE_EX_SEED_TABLE`、
`PUYO_LIVE_EX_NATIVE_DIR` で差し替えられる。24hの配布では省略時にapp/models/live_exと同梱実行系を使う。
配信専用processで正式nativeの指紋とpolicy/種表のSHAを検査する。
学習側 `1c2e2ae9` の116本を土台に、retR本体 `5c02afed` と続き `7ae53bf7` の3ファイルを取り込んだ。
計117本（解除事象の監視を含む）を専用名前空間へ固定し、import先以外は変更していない。
`_live_ex_frozen/SOURCE.json` に元と変換後のSHAを保存する。

仕事量は `WORK_LIMIT=450`。親とEX子に `PUYO_EX_WORK_BUDGET=1`、`PUYO_EX_WORK_LIMIT=450` を渡す。
`LOGICAL_DELAY_MS_DEFAULT=200` と保持0.5秒はPC2のL200Hで確定した試験値。
要求時刻は親のsubmitの単調時計。採用可能時刻は `max(要求時刻+D, IPC到着時刻)`。
早く終わった要求はDまで待ち、遅着した要求は到着後に候補となる。
利用先B3行のcapture時刻までに採用可能である必要もあり、計算待ち中に完成した未来の情報を古い行へ遡って足さない。
世代・試合・連鎖キー、入力の未来参照禁止、入力から500ms以内という従来の門を維持する。
新しく採る結果は壁時計・媒体時刻とも入力から500ms以内に限る。
通信の欠測時は学習と同じHoldPolicyで直前の採用を最大0.5秒保持する。元入力からの上限は1秒。
学習と同じEventMonitorの発火・着地・おじゃま落下・境界・対象変更で側別に解除する。
試合外・世代変更・計算失敗は保持しない。部分欠測の列を個別に過去値で埋める処理は加えない。
この500msの扱いは学習再結合のFRESH_SECと同じで、Dによる待機とは別の条件である。

`live_24h.jsonl` に起動設定、観測鍵、要求/到着/採用可能時刻、結果、推論に渡した40列、選択元を記録する。
`b3_config.json` では実際の推論モデルと、特徴計算のための旧モデルを区別する。
モデル例外は既存B3失敗経路へ返す。EX子の失敗は監査に残し、同じC2+EXモデルの欠測入力として扱う。

検証範囲はジョブ `codex_jobs/live_24h_20261027` のRESULT_h*.mdを参照。
通常入口は24h、Start_Auto_Sequence_24g2like.cmdは24g2相当。両方とも取り込みスレッドはOFF。
2試合の助走付きF/R3は確認したが、実OBSの長時間運転と採否は別途判断する。
