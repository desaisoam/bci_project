#!/usr/bin/env python3
"""
Convenience runner for the BrainFlow playback demo.
"""
import os
import sys
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    cmd = [
        sys.executable, "-u",
        os.path.join(REPO_ROOT, "scripts", "run_model.py"),
        "--model", "exp/demo_openbci_playback",
        "--save", "True",
        "--logfile", "True",
    ]
    print("Launching demo:")
    print(" ", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()

