$ErrorActionPreference='Stop'
$work='D:/puyo_analyzer/packaging/build_20261021c_work'
$repo='<work>/wt_live_v49f'
$utf8=[Text.UTF8Encoding]::new($false)
$python='<work>/build_20261019a_stage/PuyoLive/python/python.exe'
Set-Location -LiteralPath $repo

function Save-Progress([string]$Step) {
    [IO.File]::WriteAllText("$work/continuation_progress.json",
        (@{at=(Get-Date).ToString('o');step=$Step;pid=$PID}|ConvertTo-Json),$utf8)
}

function Invoke-Development([string]$Step, [string[]]$PythonArgs) {
    # 検証・組立だけを省電力4論理CPUへ制限し、正式測定には継承させない。
    $literals=($PythonArgs | ForEach-Object {"'"+$_.Replace("'","''")+"'"}) -join ','
    $script=@"
`$p=Get-Process -Id `$PID
`$p.ProcessorAffinity=[IntPtr]61440
`$p.PriorityClass='BelowNormal'
`$env:TEMP='$work/temp'
`$env:TMP=`$env:TEMP
`$env:PYTHONIOENCODING='utf-8'
Set-Location -LiteralPath '$repo'
& '$python' @($literals) *> '$work/$Step.log'
exit `$LASTEXITCODE
"@
    $path="$work/$Step.ps1"
    [IO.File]::WriteAllText($path,$script,$utf8)
    Save-Progress $Step
    $child=Start-Process powershell -WindowStyle Hidden -PassThru -Wait -ArgumentList @('-NoProfile','-File',$path)
    if ($child.ExitCode -ne 0) { throw "$Step failed: $($child.ExitCode)" }
}

try {
    $parent=Get-Process -Id $PID
    if ($parent.ProcessorAffinity.ToInt64() -ne 65535 -or [string]$parent.PriorityClass -ne 'Normal') {
        throw '正式測定の親が既定CPU・Normalではない'
    }
    Save-Progress 'waiting_for_full_parity'
    while (-not (Test-Path -LiteralPath "$work/parity_final/replay_candidate/receipt.json")) {
        Start-Sleep -Seconds 60
        $running=@(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'run_obs_recognition_parity_20261021.ps1'
        })
        if ($running.Count -eq 0 -and -not (Test-Path -LiteralPath "$work/parity_final/replay_candidate/receipt.json")) {
            throw '全行比較用の再生が未完了のまま停止'
        }
    }
    Invoke-Development 'compare_final' @('-B','packaging/compare_live_recognition_20261021.py',"$work/parity_final")
    $parity=Get-Content -LiteralPath "$work/parity_final/recognition_parity.json" -Raw -Encoding utf8 | ConvertFrom-Json
    if ($parity.rows -ne 5921 -or $parity.differing_rows -ne 0) { throw '全5921行の一致未確認' }
    Invoke-Development 'build_refresh' @('-B','packaging/build_live_obs_20261021c.py','refresh')
    Invoke-Development 'build_seal' @('-B','packaging/build_live_obs_20261021c.py','seal')
    Invoke-Development 'bundle_compare' @('-B','packaging/verify_obs_21c_bundle.py')
    Save-Progress 'formal_abba'
    & powershell -NoProfile -File packaging/start_obs_abba_20261021c.ps1 *> "$work/formal_controller.log"
    if ($LASTEXITCODE -ne 0) { throw '正式ABBA controller failed' }
    $abba='D:/puyo_analyzer/packaging/abba_20261021_bc'
    if (Test-Path -LiteralPath "$abba/measurement_deferred.json") {
        Save-Progress 'sealed_measurement_deferred'
        exit 0
    }
    Invoke-Development 'analyze_abba' @('-B','packaging/analyze_obs_abba.py','--root',$abba,
        '--runs','A1_b','B1_c','B2_c','A2_b','--labels','21b','21c','--candidate-label','21c')
    Save-Progress 'analysis_complete'
} catch {
    [IO.File]::WriteAllText("$work/continuation_failure.json",
        (@{at=(Get-Date).ToString('o');error=$_.Exception.Message}|ConvertTo-Json),$utf8)
    throw
}
