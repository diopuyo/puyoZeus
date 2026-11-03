param([switch]$SkipSmoke)
$ErrorActionPreference='Stop'
$work='D:/puyo_analyzer/packaging/build_20261021b_work'
$abba='D:/puyo_analyzer/packaging/abba_20261021_default_r2'
$bundle='D:/puyo_analyzer/packaging/build_20261021b_stage/PuyoLive'
$python='<work>/build_20261019a_stage/PuyoLive/python/python.exe'
$env:TEMP="$work/temp"
$env:TMP=$env:TEMP
$env:PYTHONIOENCODING='utf-8'
$utf8=[Text.UTF8Encoding]::new($false)
Set-Location <work>/wt_live_v49f
try {
    if (-not $SkipSmoke) {
        $ErrorActionPreference='Continue'
        & $python -B -u packaging/smoke_live_obs_20261020.py --bundle $bundle --out "$work/zip_smoke" --verified-runtime <work>/build_20261019a_stage/PuyoLive/python --duration 240 --startup-timeout 1800 --video C:/puyo1_tools/soak/media/video_d04.mp4 --start-sec 600 --ffmpeg-dll-dir <work>/media_tmp/b2_cv_20261020/cv2 --port 18854 --fast-audit --exact-seed --stdlib-cache *> "$work/zip_smoke.log"
        $result=$LASTEXITCODE
        $ErrorActionPreference='Stop'
        if ($result -ne 0) { throw "smoke failed: $result" }
        & node packaging/check_live_obs_dom.cjs "$work/zip_smoke" *> "$work/dom.log"
        if ($LASTEXITCODE -ne 0) { throw 'DOM check failed' }
    }
    # 実測中は配布と同じ起動元設定。監視器もaffinity・優先度を操作しない。
    $observer=Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList @(
        '-NoProfile','-File','<work>/wt_live_v49f/packaging/monitor_obs_abba.ps1',
        '-Output',"$abba/environment_r2.jsonl",'-StopFile',"$abba/monitor.stop")
    $ErrorActionPreference='Continue'
    & powershell -NoProfile -File packaging/run_obs_abba_20261021.ps1 *> "$abba/controller.log"
    $result=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    if ($result -ne 0) { throw "ABBA controller failed: $result" }
    [IO.File]::WriteAllText("$work/default_validation.exit.json",
        (@{exit_code=0;at=(Get-Date).ToString('o')}|ConvertTo-Json),$utf8)
} catch {
    [IO.File]::WriteAllText("$work/default_validation.failure.json",
        (@{error=$_.Exception.Message;at=(Get-Date).ToString('o')}|ConvertTo-Json),$utf8)
    throw
} finally {
    [IO.File]::WriteAllText("$abba/monitor.stop",(Get-Date).ToString('o'),$utf8)
}
