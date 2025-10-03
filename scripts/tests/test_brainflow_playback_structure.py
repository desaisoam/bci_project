#!/usr/bin/env python3
"""
Unit check: verify BrainFlow playback data shape and counters for OpenBCI CytonDaisy.
Requires brainflow (`pip install brainflow`). Uses included cyton_daisy_test_data.csv.
"""
import os
import time
import numpy as np

def main():
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
        from brainflow.data_filter import DataFilter
    except Exception as e:
        print("❌ BrainFlow not available. Install with: pip install brainflow")
        raise

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    playback_file = os.path.join(repo, "cyton_daisy_test_data.csv")
    if not os.path.isfile(playback_file):
        raise FileNotFoundError(f"Playback CSV not found at {playback_file}")

    params = BrainFlowInputParams()
    params.file = playback_file
    params.master_board = BoardIds.CYTON_DAISY_BOARD

    board = BoardShim(BoardIds.PLAYBACK_FILE_BOARD, params)

    master_id = BoardIds.CYTON_DAISY_BOARD
    sampling_rate = BoardShim.get_sampling_rate(master_id)
    package_num_channel = BoardShim.get_package_num_channel(master_id)
    eeg_channels = BoardShim.get_eeg_channels(master_id)
    timestamp_channel = BoardShim.get_timestamp_channel(master_id)

    print("Sampling rate:", sampling_rate)
    print("EEG channels:", eeg_channels)
    print("Package counter channel:", package_num_channel)
    print("Timestamp channel:", timestamp_channel)

    board.prepare_session()
    board.config_board("loopback_false")  # don't loop file
    board.start_stream()

    time.sleep(0.1)  # let some data accrue

    data_2d = board.get_board_data()
    board.stop_stream()
    board.release_session()

    print("Data shape [channels x samples] =", data_2d.shape)
    assert data_2d.ndim == 2 and data_2d.shape[0] > 0 and data_2d.shape[1] > 0

    eeg_data = data_2d[eeg_channels, :].T
    sample_counters = data_2d[package_num_channel, :]
    brainflow_ts = data_2d[timestamp_channel, :]

    print("EEG window shape [samples x 16] =", eeg_data.shape)
    print("First sample counters:", sample_counters[:5].astype(int))
    print("First timestamps:", brainflow_ts[:3])

    assert eeg_data.shape[1] == 16, "Expected 16 EEG channels"
    assert np.all(np.diff(sample_counters[:10]) >= 0), "Sample counters should be non-decreasing"
    print("✅ BrainFlow playback structure PASS")

if __name__ == "__main__":
    main()

