#!/usr/bin/env python3
# VRAM 피크 폴러(1Hz) — 사용법: python3 vram_poll.py <duration_sec> > vram.log
import subprocess
import sys
import time


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0
    peak = 0
    t0 = time.time()
    while time.time() - t0 < dur:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        try:
            mib = int(r.stdout.strip().split("\n")[0])
            peak = max(peak, mib)
            print(f"SAMPLE {mib}", flush=True)
        except (ValueError, IndexError):
            pass
        time.sleep(1)
    print(f"PEAK_VRAM_MIB={peak}")


if __name__ == "__main__":
    main()
