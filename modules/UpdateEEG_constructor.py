import numpy as np
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

print('#'*50)
print('INITIALIZING OpenBCI EEG...')

# OpenBCI CytonDaisy configuration
nChannels = 17  # 16 EEG channels + 1 sample counter (total buffer columns)
board_id = BoardIds.CYTON_DAISY_BOARD  # = 2

# Buffer management (keep existing logic)
bufferoffset = eegbuffersignal.shape[0] // 2
bufferInd = bufferoffset
eegbufferindex[:] = bufferInd

prevCount = 0  # used to test for missing data
totalSamples = 0
samplesToSkip = 0  # number of samples to skip because of filtering

# BrainFlow setup (replaces TCP socket setup)
brainflow_params = BrainFlowInputParams()
brainflow_params.serial_port = params.get('serial_port', '/dev/ttyUSB0')  # From YAML params

# Create and prepare board
board = BoardShim(board_id, brainflow_params)
board.prepare_session()
board.start_stream()

# Get sampling rate and channel info
sampling_rate = BoardShim.get_sampling_rate(board_id)  # Should be 125 Hz
eeg_channels = BoardShim.get_eeg_channels(board_id)   # Should be [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
package_num_channel = BoardShim.get_package_num_channel(board_id)  # Should be 0 - contains real sample counter

totalValidEEGSamples[:] = 0
t0 = time.time()
tickNo = 0

print(f'BrainFlow OpenBCI ready! Sampling rate: {sampling_rate} Hz, EEG channels: {len(eeg_channels)}, Package counter: channel {package_num_channel}')
