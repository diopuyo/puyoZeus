param([string]$Output = 'D:/puyo_analyzer/b3_data/codex_tasks/dist_app_release')
$ErrorActionPreference = 'Stop'
python "$PSScriptRoot/build_release.py" --output $Output
exit $LASTEXITCODE
