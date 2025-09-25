#!/usr/bin/env python3
"""
Dry wiring check for a RASPy model YAML.

Prints:
  - Modules discovered (yaml key -> file name)
  - group_params targets and keys applied
  - Sync wiring (upstream dependencies per module)
  - Triggered roots (modules with no sync or sync=None)
  - Basic validation of references (unknown module names in sync/groups)

Usage:
  python scripts/verify_model.py --model exp/demo_openbci_playback
"""
import argparse
import os
import sys
import yaml
try:
    from yaml import CLoader as Loader
except Exception:
    from yaml import Loader


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model path under models/ without .yaml (e.g., exp/kf-8-directions-gaze)")
    args = ap.parse_args()

    yaml_path = os.path.join(REPO_ROOT, "models", f"{args.model}.yaml")
    if not os.path.isfile(yaml_path):
        print(f"Error: YAML not found: {yaml_path}")
        sys.exit(2)

    with open(yaml_path, "r") as f:
        y = yaml.load(f, Loader=Loader)

    signals = y.get("signals", {})
    modules = y.get("modules", {})
    group_params = y.get("group_params", {})

    print("=== Model:", args.model)
    print("YAML:", yaml_path)
    print("Signals:", len(signals))
    print("Modules:", len(modules))

    # Modules overview
    print("\n-- Modules (yaml_key -> file name)")
    for key, mod in modules.items():
        name = mod.get("name", key)
        print(f"  {key} -> {name}")

    # group_params targets
    if group_params:
        print("\n-- group_params targets")
        mod_names = set(modules.keys())
        for gname, grp in group_params.items():
            if not isinstance(grp, dict) or "params" not in grp:
                print(f"  {gname}: (skipped, missing params)")
                continue
            if gname == "global":
                targets = list(mod_names)
            else:
                targets = grp.get("modules", []) or []
            invalid = [m for m in targets if m not in mod_names]
            print(f"  {gname}: targets={targets if len(targets)<=8 else targets[:8]+['...']} params={list(grp['params'].keys())}")
            if invalid:
                print(f"    ! invalid module refs: {invalid}")
    else:
        print("\n-- group_params: none")

    # Sync wiring
    print("\n-- Sync wiring (module waits for -> [upstream])")
    for key, mod in modules.items():
        sync_list = mod.get("sync", None)
        if sync_list is None:
            sync_list = None
        else:
            # normalize
            if isinstance(sync_list, (list, tuple)):
                sync_list = list(sync_list)
            else:
                sync_list = [sync_list]
        print(f"  {key}: {sync_list}")

    # Trigger roots
    roots = [k for k, m in modules.items() if ("sync" not in m) or (m.get("sync") is None)]
    print("\n-- Triggered roots (no sync / sync: null):", roots)

    # Validate sync references
    print("\n-- Validation")
    all_names = set(modules.keys())
    bad = {}
    for key, mod in modules.items():
        sync_list = mod.get("sync", None)
        if sync_list is None:
            continue
        if not isinstance(sync_list, (list, tuple)):
            sync_list = [sync_list]
        invalid = [s for s in sync_list if s not in all_names]
        if invalid:
            bad[key] = invalid
    if bad:
        print("  ! Unknown sync references:")
        for k, inv in bad.items():
            print(f"    {k}: {inv}")
    else:
        print("  OK: all sync references resolve")

    print("\nDone.")


if __name__ == "__main__":
    main()

