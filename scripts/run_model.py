#!/usr/bin/env python3
"""
Simple runner for RASPy models.

Examples:
  - Live (OpenBCI/BrainFlow):
      python scripts/run_model.py --model exp/kf-8-directions-gaze

  - Choose save folder and toggle logfile:
      python scripts/run_model.py --model exp/kf-8-directions-gaze \
          --data_folder my_run_{date}_{counter} --logfile True --save True

Notes:
  - This is a thin wrapper around main/main.py to make launching consistent.
  - It does not modify YAMLs; ensure ports/paths in your YAML are valid.
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(REPO_ROOT)  # move up from scripts/


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="exp/kf-8-directions-gaze",
                        help="Model YAML under models/ without extension (e.g., exp/kf-8-directions-gaze)")
    parser.add_argument("--save", default="True", choices=["True", "False", "true", "false", "0", "1"],
                        help="Whether to save run artifacts (overrides logfile when False)")
    parser.add_argument("--logfile", default="True", choices=["True", "False", "true", "false", "0", "1"],
                        help="Whether to write a logfile when saving is enabled")
    parser.add_argument("--data_folder", default=None,
                        help="Custom relative folder name under configured save_path; supports {date} and {counter}")
    parser.add_argument("--module_args", default="",
                        help="Extra module args string passed through to main (advanced)")
    parser.add_argument("-overwrite_params", default=False, action="store_true",
                        help="If set, module_args also overwrite params (not just commandline_args)")
    args = parser.parse_args()

    # Validate model file exists
    yaml_path = os.path.join(REPO_ROOT, "models", f"{args.model}.yaml")
    if not os.path.isfile(yaml_path):
        print(f"Error: YAML not found: {yaml_path}")
        sys.exit(2)

    # Build command
    cmd = [
        sys.executable, "-u",
        os.path.join(REPO_ROOT, "main", "main.py"),
        args.model,
        "--save", args.save,
        "--logfile", args.logfile,
        "--module_args", args.module_args,
    ]
    if args.overwrite_params:
        cmd.append("-overwrite_params")
    if args.data_folder is not None:
        cmd.extend(["--data_folder", args.data_folder])

    print("Launching:")
    print(" ", " ".join(cmd))
    print("")

    # Run from repo root to match main.py expectations
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()

    # Surface common failure due to known group_params issue for quick diagnosis
    if proc.returncode != 0:
        print(f"main exited with code {proc.returncode}")
        print("If you see a TypeError about 'dict_keys' being not subscriptable,\n"
              "it is caused by group_params handling in main/main.py.\n"
              "We can fix that separately.")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
