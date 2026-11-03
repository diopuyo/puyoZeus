$ErrorActionPreference='Stop'
$current=Get-Process -Id $PID
$current.ProcessorAffinity=[IntPtr]61440
$current.PriorityClass='BelowNormal'
$work='D:/puyo_analyzer/packaging/build_20261021c_work/parity_final'
New-Item -ItemType Directory -Force "$work/temp" | Out-Null
$env:TEMP="$work/temp"
$env:TMP=$env:TEMP
$env:PYTHONIOENCODING='utf-8'
$env:OPENCV_FFMPEG_DLL_DIR='<work>/media_tmp/b2_cv_20261020/cv2'
Set-Location <work>/wt_live_v49f
$python='<work>/build_20261019a_stage/PuyoLive/python/python.exe'
foreach ($version in @('baseline','candidate')) {
    $argsRun=@('-B','-u','packaging/replay_live_recognition_20261021.py',
        '--source','D:/puyo_analyzer/packaging/abba_20261021_default_r2/A2_f_fixed/output',
        '--out',"$work/replay_$version")
    if ($version -eq 'candidate') { $argsRun+='--candidate'; $env:PUYO_OBS_PREPARE_EXACT='1' }
    $ErrorActionPreference='Continue'
    & $python @argsRun *> "$work/replay_$version.log"
    $result=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    if ($result -ne 0) { throw "replay $version failed: $result" }
}
