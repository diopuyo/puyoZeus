"""遅延測定と劣化検証を重ねず、順に実行する。"""
from pathlib import Path
import subprocess
import sys


def main() -> None:
    jobs = [
        ('live_b4_realtime_verified', ['scripts.run_live_pipeline_20260928', '--realtime',
                                     '--coalesce-features', '--output', 'logs/live_b4_realtime_verified']),
        ('live_b4_degradation', ['scripts.verify_live_b4_degradation']),
    ]
    for name, arguments in jobs:
        destination = Path('logs') / name
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / 'run.log').open('w') as log:
            result = subprocess.run([sys.executable, '-u', '-m', *arguments],
                                    stdout=log, stderr=subprocess.STDOUT)
        (destination / 'exitcode.txt').write_text(str(result.returncode))
        if result.returncode:
            raise SystemExit(result.returncode)
    subprocess.run([sys.executable, '-m', 'scripts.summarize_live_b4_20260928'], check=True)


if __name__ == '__main__':
    main()
