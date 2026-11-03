$ErrorActionPreference='Stop'
$work='D:/puyo_analyzer/packaging/abba_20261021_bc'
$utf8=[Text.UTF8Encoding]::new($false)
New-Item -ItemType Directory -Force "$work/temp" | Out-Null
Set-Location -LiteralPath '<work>/wt_live_v49f'
$protocol=@{order=@('A1_b','B1_c','B2_c','A2_b');versions=@('21b','21c','21c','21b');
    video='C:/puyo1_tools/soak/media/video_d04.mp4';start_sec=0;end_sec=1200;
    ratio=1.647385423;affinity='inherited default';priority='inherited default';
    quiet_check_interval_sec=300;quiet_deadline='2026-10-21T19:30:00+09:00';
    first_publication='state.publish後のmonotonic。latest frameとB3元入力を別集計';
    output_preserving_changes=@('fixed HSV native','fixed ONNX model root cache','startup preparations');
    candidate_flags=@('PUYO_OBS_PREPARE_EXACT=1','PUYO_OBS_DEEP_TIMING=1')}
[IO.File]::WriteAllText("$work/protocol.json",($protocol|ConvertTo-Json -Depth 6),$utf8)
$observer=Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList @(
    '-NoProfile','-File','<work>/wt_live_v49f/packaging/monitor_obs_abba.ps1',
    '-Output',"$work/environment_r2.jsonl",'-StopFile',"$work/monitor.stop")
try {
    $ErrorActionPreference='Continue'
    & powershell -NoProfile -File packaging/run_obs_abba_20261021c.ps1 *> "$work/controller.log"
    $result=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    if ($result -ne 0) { throw "ABBA controller exit=$result" }
} catch {
    [IO.File]::WriteAllText("$work/controller.failure.json",
        (@{at=(Get-Date).ToString('o');error=$_.Exception.Message}|ConvertTo-Json),$utf8)
    throw
} finally {
    [IO.File]::WriteAllText("$work/monitor.stop",(Get-Date).ToString('o'),$utf8)
}
