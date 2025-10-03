import numpy as np

params = globals().get('params', {})
state_task = globals()['state_task']
decoded_pos = globals()['decoded_pos']
target_pos = globals()['target_pos']
game_state = globals()['game_state']

try:
    _internal
except NameError:
    _internal = {
        'initialized': False,
        'state_idx': 0,
        'ticks_left': 0,
        'seq': [0, 1, 2, 3],
        'hold_ticks': 400,
        'amplitude': 0.7,
        'last_printed': None,
    }

if not _internal['initialized']:
    seq = params.get('state_sequence', _internal['seq'])
    _internal['seq'] = list(seq)
    _internal['state_idx'] = 0
    _internal['hold_ticks'] = int(params.get('hold_ticks', _internal['hold_ticks']))
    _internal['ticks_left'] = _internal['hold_ticks']
    _internal['amplitude'] = float(params.get('amplitude', _internal['amplitude']))
    # Initialize outputs
    state_task[:] = np.int8(0)
    decoded_pos[:] = np.float32(0.0)
    target_pos[:] = np.float32(0.0)
    game_state[:] = np.int8(0)
    _internal['initialized'] = True
    _internal['last_printed'] = None

# Advance state if hold count elapsed
_internal['ticks_left'] -= 1
if _internal['ticks_left'] <= 0:
    _internal['state_idx'] = (_internal['state_idx'] + 1) % len(_internal['seq'])
    _internal['ticks_left'] = _internal['hold_ticks']

state = int(_internal['seq'][_internal['state_idx']])
amp = _internal['amplitude']

mapping = {
    0: np.array([-amp, 0.0], dtype=np.float32),
    1: np.array([ amp, 0.0], dtype=np.float32),
    2: np.array([0.0,  amp], dtype=np.float32),
    3: np.array([0.0, -amp], dtype=np.float32),
    4: np.array([0.0, 0.0], dtype=np.float32),
}
tgt = mapping.get(state, np.array([0.0, 0.0], dtype=np.float32))

state_task[0] = np.int8(state)
target_pos[:] = tgt
decoded_pos[:] = 0.8 * decoded_pos + 0.2 * tgt
game_state[0] = np.int8(1)

if _internal.get('last_printed') != state:
    print(f"dummy_task state -> {state}, target {tgt}")
    _internal['last_printed'] = state
