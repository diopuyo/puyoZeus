param([string]$Root='D:/puyo_analyzer/packaging/abba_20261021_bc')
$ErrorActionPreference='Stop'

function Read-LastRecord([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $lines=@(Get-Content -LiteralPath $Path -Tail 2 -Encoding utf8)
    for ($index=$lines.Count-1; $index -ge 0; $index--) {
        try { return ($lines[$index] | ConvertFrom-Json) } catch { }
    }
    return $null
}

# 状態確認用Pythonの起動が静穏判定へ混入しないよう、PowerShellだけで末尾を読む。
$environment=Read-LastRecord "$Root/environment_r2.jsonl"
$run=$environment.active_run
$recognition=Read-LastRecord "$Root/$run/output/spool/frame_timing_recognition.jsonl"
$poll=Read-LastRecord "$Root/$run/polls.jsonl"
$values=$poll.snapshot.evaluations
$idle=Read-LastRecord "$Root/idle_checks.jsonl"
$exits=@(Get-ChildItem -LiteralPath $Root -Filter '*.exit.json' | Select-Object -ExpandProperty Name)
@{at=$environment.at;run=$run;t_sec=$recognition.t_sec;free_gib=$environment.free_gib;
    values=@{b3=$values.practical.p1_win_probability;v49f=$values.v49f.p1;ama=$values.ama_plain.p1};
    reasons=@{v49f=$values.v49f.reason;ama=$values.ama_plain.reason};
    idle_at=$idle.at;idle_busy=$idle.busy;exits=$exits} | ConvertTo-Json -Depth 4
