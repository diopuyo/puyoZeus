"""長い比較・学習のチェックポイントで Windows PuyoLive を優先する。"""
from __future__ import annotations

import json
import subprocess
import time

CHECK_SECONDS = 20
QUERY = ("@(Get-CimInstance Win32_Process -ErrorAction Stop | "
         "Where-Object { $_.Name -match '^(PuyoLive|PuyoZeus)([-_].*)?\\.exe$' -or "
         "($_.Name -match '^python(w)?\\.exe$' -and "
         "$_.CommandLine -match 'src\\.phase_j\\.(launcher|distribution_app)') } | "
         "Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress")


class ObsPriorityGuard:
    """各作業単位を20秒以内に分けて呼ぶ。稼働中は次の単位を始めない。"""

    def __init__(self) -> None:
        self.last_check = float('-inf')

    def checkpoint(self, force: bool = False) -> None:
        if not force and time.monotonic() - self.last_check < CHECK_SECONDS:
            return
        while True:
            result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', QUERY],
                                    capture_output=True, text=True, check=True, timeout=15)
            raw = result.stdout.strip()
            pids = json.loads(raw) if raw else []
            self.last_check = time.monotonic()
            if not pids:
                return
            print(f'OBS優先: PuyoLive {pids} 稼働中。比較・学習を休止。', flush=True)
            time.sleep(CHECK_SECONDS)
