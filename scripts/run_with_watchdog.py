#!/usr/bin/env python3
"""Crash-resilient supervisor around train_la.py / train_da.py / train_ta.py.

The DGX Spark workstation has been hard-rebooting mid-training (firmware-level
power-off, no kernel logs). The underlying trainer is already configured to
save every 1000 steps to a stable directory and auto-resume from the latest
checkpoint (see configs/train/la_mlm.yaml + scripts/train_la.py); this wrapper
supervises the process: relaunch on crash, log GPU thermal/power telemetry to a
CSV, cap total restarts.

Usage:
    python scripts/run_with_watchdog.py --target la
    python scripts/run_with_watchdog.py --target da --max-restarts 20
    python scripts/run_with_watchdog.py --target ta \\
        -- train=ta_mcqa train.la_path=null train.num_choices=10

Args after `--` are passed verbatim to the underlying training script as Hydra
overrides. The watchdog itself uses normal argparse for its own flags.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

TARGETS = {
    "la": "scripts/train_la.py",
    "da": "scripts/train_da.py",
    "ta": "scripts/train_ta.py",
}


def gpu_sampler(csv_path: Path, stop: Event, interval: float = 5.0) -> None:
    """Append a row of nvidia-smi metrics every `interval` seconds.

    Captures temperature, power, GPU utilization, memory. Tolerates transient
    nvidia-smi failures (writes an ERR row and keeps going).
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp_utc", "temperature.gpu", "power.draw",
              "utilization.gpu", "memory.used"]
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(fields)
            f.flush()
        while not stop.is_set():
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=" + ",".join(fields[1:]),
                    "--format=csv,noheader,nounits",
                ], text=True, timeout=4).strip().split(",")
                writer.writerow([int(time.time()), *[v.strip() for v in out]])
            except Exception as e:
                writer.writerow([int(time.time()), "ERR", str(e)[:80], "", ""])
            f.flush()
            stop.wait(interval)


def run_once(cmd: list[str], log_path: Path) -> int:
    """Launch the trainer as a child process; append stdout/stderr to log_path.

    Uses preexec_fn=os.setsid so the child becomes its own process-group
    leader; that way a Ctrl-C in the supervisor cleanly kills the whole
    training subtree (Python + DataLoader workers).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write(f"\n=== launch @ {time.ctime()} :: {' '.join(cmd)} ===\n".encode())
        log.flush()
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid,
        )
        try:
            return proc.wait()
        except KeyboardInterrupt:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            return proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, choices=sorted(TARGETS),
                    help="which training script to supervise")
    ap.add_argument("--max-restarts", type=int, default=20,
                    help="cap on relaunch attempts after non-zero exits")
    ap.add_argument("--restart-wait", type=int, default=60,
                    help="seconds to wait between relaunches (GPU cool-down)")
    ap.add_argument("--log-dir", type=Path, default=Path("/tmp/watchdog"))
    ap.add_argument("--gpu-interval", type=float, default=5.0,
                    help="nvidia-smi sampling cadence in seconds")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="Hydra overrides for the inner trainer (after `--`)")
    args = ap.parse_args()

    train_script = TARGETS[args.target]
    if not Path(train_script).exists():
        sys.exit(f"[watchdog] missing trainer at {train_script}")

    overrides = [a for a in args.rest if a != "--"]

    log_path = args.log_dir / f"{args.target}.log"
    gpu_csv = args.log_dir / f"{args.target}_gpu.csv"
    args.log_dir.mkdir(parents=True, exist_ok=True)

    stop = Event()
    sampler = Thread(
        target=gpu_sampler, args=(gpu_csv, stop, args.gpu_interval), daemon=True,
    )
    sampler.start()

    cmd = [sys.executable, "-u", train_script, *overrides]
    print(f"[watchdog] cmd = {' '.join(cmd)}", flush=True)
    print(f"[watchdog] log = {log_path}", flush=True)
    print(f"[watchdog] gpu = {gpu_csv}", flush=True)
    print(f"[watchdog] max_restarts={args.max_restarts}, "
          f"restart_wait={args.restart_wait}s", flush=True)

    success = False
    for attempt in range(1, args.max_restarts + 1):
        print(f"[watchdog] attempt {attempt}/{args.max_restarts} @ {time.ctime()}",
              flush=True)
        rc = run_once(cmd, log_path)
        print(f"[watchdog] child exited rc={rc} @ {time.ctime()}", flush=True)
        if rc == 0:
            print("[watchdog] success — child returned 0", flush=True)
            success = True
            break
        if attempt < args.max_restarts:
            print(f"[watchdog] sleeping {args.restart_wait}s before relaunch", flush=True)
            time.sleep(args.restart_wait)
    if not success:
        print(f"[watchdog] gave up after {args.max_restarts} attempts", flush=True)

    stop.set()
    sampler.join(timeout=10)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
