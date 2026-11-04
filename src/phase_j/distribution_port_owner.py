"""応答しないWindowsの待受を、同じ利用者と配布物の目印で確認する。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

QUERY_TIMEOUT = 15
MARKER = 'app/src/phase_j/distribution_app.py'
OWNER_SCRIPT = r'''
$ErrorActionPreference='Stop'
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding $false
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$request=[Console]::In.ReadToEnd() | ConvertFrom-Json
$rows=@(Get-NetTCPConnection -LocalPort $request.port -State Listen |
    Where-Object {$_.LocalAddress -in @('127.0.0.1','0.0.0.0')})
if($rows.Count -ne 1) {exit 1}
$ownerId=[int]$rows[0].OwningProcess
$process=[System.Diagnostics.Process]::GetProcessById($ownerId)
# ハンドルを先に保持し、PIDの再利用後の別プロセスを終了させない。
$handle=$process.Handle
$entry=Get-CimInstance Win32_Process -Filter "ProcessId=$ownerId"
$sid=(Invoke-CimMethod -InputObject $entry -MethodName GetOwnerSid).Sid
if($sid -ne [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value) {exit 1}
$identity=@{pid=$ownerId; executable=$process.MainModule.FileName;
    created=$process.StartTime.ToUniversalTime().Ticks.ToString()}
if($request.stop) {
    if($identity.pid -ne $request.pid -or $identity.executable -cne $request.executable -or
       $identity.created -ne $request.created) {exit 1}
    $parent=Split-Path (Split-Path $identity.executable)
    if(!(Test-Path -LiteralPath (Join-Path $parent 'app/src/phase_j/distribution_app.py') -PathType Leaf)) {exit 1}
    $process.Kill()
}
$identity | ConvertTo-Json -Compress
'''


def query_owner(port: int, expected: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if os.name != 'nt':
        return None
    request = dict(port=port, stop=expected is not None)
    request.update(expected or {})
    try:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', OWNER_SCRIPT],
                                input=json.dumps(request), capture_output=True, encoding='utf-8',
                                timeout=QUERY_TIMEOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        return json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def inspect_owner(port: int) -> dict[str, Any] | None:
    owner = query_owner(port)
    if not isinstance(owner, dict) or type(owner.get('pid')) is not int:
        return None
    executable = Path(owner.get('executable', ''))
    root = executable.parent.parent
    if (executable.name.lower() != 'pythonw.exe' or executable.parent.name.lower() != 'python'
            or not executable.is_absolute() or not (root / MARKER).is_file()):
        return None
    return dict(kind='unresponsive', port=port, root=str(root), **owner)


def stop_owner(conflict: dict[str, Any]) -> None:
    if inspect_owner(conflict['port']) != conflict:
        raise ValueError('ポートの持ち主が変わりました / Port owner changed')
    expected = {key: conflict[key] for key in ('pid', 'executable', 'created')}
    if query_owner(conflict['port'], expected) != expected:
        raise ValueError('同じ利用者のアプリを終了できません / Could not stop the verified app')
