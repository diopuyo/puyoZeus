# 完了ファイルだけを読み、実行を起動しない。
$v5Root = 'D:/puyo_analyzer/wt_prefire/logs/prefire_prediction'
$v5Done = @(Get-ChildItem "$v5Root/replay/v5_L03" -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName 'DONE.json') })
$v5Short = $null
if (Test-Path "$v5Root/v5_short/result.json") {
    $v5Short = Get-Content -Raw "$v5Root/v5_short/result.json" | ConvertFrom-Json
}
[PSCustomObject]@{
    done = @($v5Done | ForEach-Object { $_.Name })
    ledger = (Test-Path "$v5Root/v5_experiment/samples_ledger.json")
    experiment = (Test-Path "$v5Root/v5_experiment/ledger/summary.json")
    short_v3 = ($null -ne $v5Short -and $v5Short.model_dir -eq 'models/exchange_event_v3')
    errors = @(Get-ChildItem "$v5Root/replay/v5_L03_*.log", "$v5Root/v5_experiment/ledger.log", "$v5Root/v5_experiment/ledger_experiment.log" -ErrorAction SilentlyContinue | Select-String -SimpleMatch 'Traceback').Count
} | ConvertTo-Json -Compress
