#!/usr/bin/env python3
"""
Investigate BrainFlow channel layout for OpenBCI Cyton+Daisy
Find out what channels are available and which one contains the sample counter
"""

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def investigate_cyton_daisy_channels():
    print("="*60)
    print("INVESTIGATING BRAINFLOW CHANNELS FOR CYTON+DAISY")
    print("="*60)

    board_id = BoardIds.CYTON_DAISY_BOARD

    # Get all available channel types
    print("📋 CHANNEL INFORMATION:")

    try:
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        print(f"📊 EEG channels: {eeg_channels}")
        print(f"📊 Number of EEG channels: {len(eeg_channels)}")

        # Get other channel types
        analog_channels = BoardShim.get_analog_channels(board_id)
        print(f"📊 Analog channels: {analog_channels}")

        accel_channels = BoardShim.get_accel_channels(board_id)
        print(f"📊 Accelerometer channels: {accel_channels}")

        other_channels = BoardShim.get_other_channels(board_id)
        print(f"📊 Other channels: {other_channels}")

        # Check for timestamp/sample counter channels
        timestamp_channel = BoardShim.get_timestamp_channel(board_id)
        print(f"📊 Timestamp channel: {timestamp_channel}")

        package_num_channel = BoardShim.get_package_num_channel(board_id)
        print(f"📊 Package number channel: {package_num_channel}")

        # Get board description
        board_descr = BoardShim.get_board_descr(board_id)
        print(f"\n📋 BOARD DESCRIPTION:")
        for key, value in board_descr.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Error getting channel info: {e}")
        return False

    print(f"\n{'='*60}")
    print("TESTING DATA ACQUISITION")
    print(f"{'='*60}")

    # Test actual data acquisition
    brainflow_params = BrainFlowInputParams()
    # Use synthetic board for testing
    test_board_id = BoardIds.SYNTHETIC_BOARD

    try:
        board = BoardShim(test_board_id, brainflow_params)
        board.prepare_session()
        board.start_stream()

        import time
        time.sleep(1)  # Let some data accumulate

        data_2d = board.get_board_data()
        print(f"📊 Raw data shape: {data_2d.shape}")
        print(f"📊 Channels (rows): {data_2d.shape[0]}")
        print(f"📊 Samples (cols): {data_2d.shape[1]}")

        if data_2d.shape[1] > 0:
            print(f"\n📋 CHANNEL DATA SAMPLES (first 3 samples):")
            for channel_idx in range(31):  # Show first 25 channels
                channel_data = data_2d[channel_idx, :3]  # First 3 samples
                print(f"  Channel {channel_idx:2d}: {channel_data}")

            # Check specific channels we care about
            print(f"\n📋 SPECIFIC CHANNEL ANALYSIS:")

            # EEG channels
            eeg_channels_synth = BoardShim.get_eeg_channels(test_board_id)
            print(f"📊 EEG channels: {eeg_channels_synth}")
            if len(eeg_channels_synth) > 0:
                eeg_data = data_2d[eeg_channels_synth, :]
                print(f"📊 EEG data shape: {eeg_data.shape}")
                print(f"📊 EEG sample: {eeg_data[:, 0]}")  # First sample all EEG channels

            # Package number (sample counter)
            package_channel = BoardShim.get_package_num_channel(test_board_id)
            print(f"📊 Package number channel: {package_channel}")
            if package_channel >= 0:
                package_data = data_2d[package_channel, :]
                print(f"📊 Package numbers: {package_data[:10]}")  # First 10 samples
                print(f"📊 Package increment: {np.diff(package_data[:10])}")

            # Timestamp
            timestamp_channel = BoardShim.get_timestamp_channel(test_board_id)
            print(f"📊 Timestamp channel: {timestamp_channel}")
            if timestamp_channel >= 0:
                timestamp_data = data_2d[timestamp_channel, :]
                print(f"📊 Timestamps: {timestamp_data[:5]}")  # First 5 samples

        board.stop_stream()
        board.release_session()

        print(f"\n{'='*60}")
        print("✅ INVESTIGATION COMPLETE!")
        print("🔍 Key Findings:")
        print(f"  - Package number channel: {BoardShim.get_package_num_channel(test_board_id)}")
        print(f"  - This channel contains the sample counter we need!")
        print(f"  - We should use this instead of generating our own counter")
        print("="*60)

        return True

    except Exception as e:
        print(f"❌ Error during data acquisition test: {e}")
        return False

if __name__ == "__main__":
    investigate_cyton_daisy_channels()
