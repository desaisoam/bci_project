#!/usr/bin/env python3
"""
Model YAML validator for RASPy.

Usage:
  python scripts/validate_model.py models/exp/open_loop_cyton.yaml

Performs static checks to catch missing params and bad signal references
before launching the full pipeline. Does NOT open hardware or spawn processes.
"""

import sys
import argparse
import yaml


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(2)


def warn(msg):
    print(f"[WARN] {msg}")


def ok(msg):
    print(f"[OK] {msg}")


def validate_signals(model):
    signals = model.get('signals', {})
    if not signals:
        fail('No signals section found')
    # Basic POSIX shm name length advisory (macOS): name gets a leading '/'
    long = [n for n in signals if len(n) >= 31]
    if long:
        warn(f'signal names may exceed macOS shm limits: {long}')
    ok(f"signals: {len(signals)} defined")
    return signals


def validate_sync(mods):
    names = set(mods.keys())
    for mname, m in mods.items():
        for up in m.get('sync', []) or []:
            if up not in names:
                fail(f"Module '{mname}' sync references unknown upstream '{up}'")
    ok('module sync graph references are valid')


def validate_io_refs(signals, mods):
    for mname, m in mods.items():
        for key in ('in', 'out'):
            for s in m.get(key, []) or []:
                if s not in signals:
                    fail(f"Module '{mname}' references unknown signal '{s}' in '{key}'")
    ok('module in/out signal references exist')


def validate_logger(signals, logger_mod):
    log = logger_mod.get('params', {}).get('log', {})
    if 'task' not in log or 'eeg' not in log:
        fail('logger.params.log must include task and eeg streams')
    for s in log['task'].get('signals', []) or []:
        if s not in signals:
            fail(f"logger task stream references unknown signal '{s}'")
    eeg_cfg = log['eeg']
    idx = eeg_cfg.get('index')
    if idx not in signals:
        fail(f"logger eeg.index '{idx}' not found in signals")
    for b in eeg_cfg.get('buffers', []) or []:
        if b not in signals:
            fail(f"logger eeg.buffers references unknown signal '{b}'")
    ok('logger stream configuration references valid signals')


def validate_updateeeg(mod):
    params = mod.get('params', {})
    if params.get('mode') == 'playback':
        if 'file' not in params:
            fail('UpdateEEG mode=playback requires params.file')
    else:
        if 'serial_port' not in params:
            warn('UpdateEEG live mode without explicit serial_port')
    cmds = params.get('openbci_commands')
    if cmds is not None and not (
        isinstance(cmds, str) or (isinstance(cmds, list) and all(isinstance(c, str) for c in cmds))
    ):
        fail('UpdateEEG.params.openbci_commands must be a string or list of strings')
    ok('UpdateEEG configuration looks sane')


def validate_sj(mod):
    params = mod.get('params', {})
    # Direct-indexed mandatory keys in initParamWithYaml
    required = [
        'sessionLength', 'screenSize', 'objScale', 'cursorRadius', 'cursorVel',
        'showAllTarget', 'ignoreWrongTarget', 'skipFirstNtrials',
        'useRandomTargetPos', 'randomTargetPosRadius', 'resetCursorPos',
        'targetsInfo', 'target2state_task', 'decodedVel',
        'holdTimeThres', 'softmaxThres', 'assist', 'assistMode',
        'activeLength', 'inactiveLength', 'yamlName', 'showSoftmax',
        'showPredictedTarget', 'kfCopilotAlpha'
    ]
    missing = [k for k in required if k not in params]
    if missing:
        fail(f"SJ_4_directions.params missing mandatory keys: {missing}")
    # targetsInfo minimal shape
    ti = params['targetsInfo']
    if not isinstance(ti, dict) or not all(k in ti for k in ('left','right','up','down')):
        fail('SJ targetsInfo must include left/right/up/down entries')
    for k, v in ti.items():
        if isinstance(v, str):
            continue
        if not (isinstance(v, list) and len(v) == 2 and len(v[0]) == 2 and len(v[1]) == 2):
            fail(f"targetsInfo['{k}'] must be [[x,y],[w,h]]")
    # decodedVel indices 0..4
    dv = params['decodedVel']
    for i in range(5):
        if i not in dv and str(i) not in dv:
            fail(f'decodedVel must include key {i}')
    # labels present
    t2s = params['target2state_task']
    for lbl in ('left','right','up','down','still'):
        if lbl not in t2s:
            fail(f"target2state_task missing '{lbl}'")
    ok('SJ_4_directions mandatory params validated')


def validate_dt(timer_mod, sj_mod):
    t_dt = timer_mod.get('params', {}).get('dt')
    s_dt = sj_mod.get('params', {}).get('dt')
    if t_dt is not None and s_dt is not None and t_dt != s_dt:
        warn(f"timer.dt ({t_dt}) != SJ.dt ({s_dt}); SJ durations may not match wall clock")
    else:
        ok('timer.dt and SJ.dt are consistent')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model_yaml', help='Path to model YAML')
    args = ap.parse_args()
    model = load_yaml(args.model_yaml)

    signals = validate_signals(model)
    modules = model.get('modules', {})
    if not modules:
        fail('No modules section found')
    for must in ('logger_disk', 'timer', 'UpdateEEG', 'filterEEG', 'SJ_4_directions', 'logger'):
        if must not in modules:
            fail(f"Required module '{must}' not found")
    ok('required modules present')

    validate_sync(modules)
    validate_io_refs(signals, modules)
    validate_logger(signals, modules['logger'])
    validate_updateeeg(modules['UpdateEEG'])
    validate_sj(modules['SJ_4_directions'])
    validate_dt(modules['timer'], modules['SJ_4_directions'])

    ok('Validation passed')
    sys.exit(0)


if __name__ == '__main__':
    main()

