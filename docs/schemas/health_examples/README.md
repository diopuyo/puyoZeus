# Phase J health canonical examples

- `starting.json`: HTTP受付開始前後の初期状態。
- `healthy.json`: snapshot配信と監査記録が正常な状態。
- `degraded.json`: telemetry障害をerror code付きで公開する状態。
- `degraded_worker.json`: telemetryは正常だが予測workerが故障した状態。
- `subscriber_limit.json`: subscriberが上限ちょうどに達した合法状態。

全例を`puyo_overlay_health_v1.schema.json`で検証する。subscriber上限、Hub latestとの一致、
degraded時のerror codeなどSchemaを跨ぐ制約は
`puyo_overlay_snapshot_v1_semantic_rules.md`のH01〜H04で検証する。
