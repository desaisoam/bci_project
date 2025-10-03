import numpy as np
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

print('#'*50)
print('INITIALIZING EEG via BrainFlow...')

# OpenBCI CytonDaisy configuration (17 cols: 16 EEG + 1 counter)
nChannels = 17

# Buffer management (keep existing logic)
bufferoffset = eegbuffersignal.shape[0] // 2
bufferInd = bufferoffset
eegbufferindex[:] = bufferInd

prevCount = 0  # for continuity checks
totalSamples = 0
samplesToSkip = 0  # number of samples to skip because of filtering

# BrainFlow setup supports live, playback (file), or synthetic
mode = params.get('mode', 'live')  # 'live' | 'playback' | 'synthetic'
brainflow_params = BrainFlowInputParams()

# Determine shim (actual board to create) and physical (channel layout) ids
if mode == 'playback':
    # Playback file required; master board defines channel mapping
    playback_file = params.get('file', None)
    if playback_file is None:
        raise ValueError('UpdateEEG: mode=playback requires params.file to be set')
    brainflow_params.file = playback_file
    # default to Cyton Daisy for mapping unless specified
    master_board = params.get('master_board', 'CYTON_DAISY_BOARD')
    # allow passing string name or numeric id
    phys_id = getattr(BoardIds, master_board) if isinstance(master_board, str) else master_board
    brainflow_params.master_board = phys_id
    shim_id = BoardIds.PLAYBACK_FILE_BOARD
elif mode == 'synthetic':
    shim_id = BoardIds.SYNTHETIC_BOARD
    phys_id = BoardIds.SYNTHETIC_BOARD
else:
    # live by default
    serial_port = params.get('serial_port', '/dev/ttyUSB0')
    brainflow_params.serial_port = serial_port
    shim_id = BoardIds.CYTON_DAISY_BOARD
    phys_id = BoardIds.CYTON_DAISY_BOARD

# Create and prepare board
board = BoardShim(shim_id, brainflow_params)
board.prepare_session()

# Optional: configure playback looping
if mode == 'playback':
    # do not loop by default unless explicitly requested
    loop = str(params.get('playback_loop', 'false')).lower() in ['1','true','yes']
    board.config_board(f"loopback_{'true' if loop else 'false'}")

# Optional: apply user-provided low-level OpenBCI/Cyton commands before starting stream
# These are board-specific ASCII commands (same as used by the OpenBCI firmware/GUI),
# passed through BrainFlow's config_board(). Use them to adjust channel gain, bias, SRB, etc.
user_cmds = params.get('openbci_commands', [])
if isinstance(user_cmds, str):
    user_cmds = [user_cmds]
for cmd in user_cmds:
    try:
        print(f"Applying board command: {cmd}")
        board.config_board(cmd)
        # small delay to avoid overloading the bridge on back-to-back commands
        time.sleep(0.05)
    except Exception as e:
        print(f"Warning: failed to apply board command '{cmd}': {e}")

board.start_stream()

# Get sampling rate and channel info from physical id
sampling_rate = BoardShim.get_sampling_rate(phys_id)
eeg_channels = BoardShim.get_eeg_channels(phys_id)
package_num_channel = BoardShim.get_package_num_channel(phys_id)
has_pkg_channel = isinstance(package_num_channel, int) and package_num_channel >= 0

# Parameters for unwrapping hardware package counter into a continuous counter
# Default modulus 256 for OpenBCI-family devices; configurable via YAML (params.pkg_modulus)
# Some devices (e.g., Cyton Daisy) increment the package counter by 2 each sample.
# You can set params.pkg_expected_step to match the device behavior (default 1) or
# disable the HW package counter entirely with params.use_pkg_counter: false.
pkg_modulus = int(params.get('pkg_modulus', 256))
pkg_expected_step = int(params.get('pkg_expected_step', 1))
use_pkg_counter = bool(params.get('use_pkg_counter', True))
pkg_epoch = 0
last_pkg = None

totalValidEEGSamples[:] = 0
t0 = time.time()
tickNo = 0

print(f"BrainFlow mode={mode}, sampling_rate={sampling_rate} Hz, EEG chans={len(eeg_channels)}, package_ch={package_num_channel}")
