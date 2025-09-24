#!/usr/bin/env python3
"""
Diagnose the timing/missing data issue
Figure out exactly what's happening with BrainFlow data timing
"""

import numpy as np
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def diagnose_timing():
    print("="*60)
    print("DIAGNOSING TIMING ISSUE")
    print("="*60)

    board_id = BoardIds.SYNTHETIC_BOARD
    brainflow_params = BrainFlowInputParams()

    package_num_channel = BoardShim.get_package_num_channel(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    print(f"📊 Sampling rate: {sampling_rate} Hz")
    print(f"📊 Expected samples per 50ms: {sampling_rate * 0.05:.1f}")

    try:
        board = BoardShim(board_id, brainflow_params)
        board.prepare_session()
        board.start_stream()

        print("\n🔍 EXPERIMENT 1: What does get_board_data() actually return?")

        # Let data accumulate for different time periods
        for wait_time in [0.01, 0.05, 0.1, 0.2]:
            print(f"\n⏱️  Waiting {wait_time*1000:.0f}ms then calling get_board_data()...")
            time.sleep(wait_time)

            data_2d = board.get_board_data()
            if data_2d.shape[1] > 0:
                sample_numbers = data_2d[package_num_channel, :]
                print(f"  📊 Got {data_2d.shape[1]} samples")
                print(f"  📊 Sample range: {sample_numbers.min():.0f} to {sample_numbers.max():.0f}")
                print(f"  📊 Expected: ~{sampling_rate * wait_time:.1f} samples")
                print(f"  📊 Actual/Expected ratio: {data_2d.shape[1] / (sampling_rate * wait_time):.2f}")
            else:
                print("  📊 No data!")

        print(f"\n🔍 EXPERIMENT 2: Rapid consecutive calls to get_board_data()")

        time.sleep(0.1)  # Let some data accumulate

        for i in range(5):
            data_2d = board.get_board_data()
            if data_2d.shape[1] > 0:
                sample_numbers = data_2d[package_num_channel, :]
                print(f"  Call {i+1}: {data_2d.shape[1]} samples, range {sample_numbers.min():.0f}-{sample_numbers.max():.0f}")
            else:
                print(f"  Call {i+1}: No data")
            time.sleep(0.001)  # Tiny delay

        print(f"\n🔍 EXPERIMENT 3: Compare get_board_data() vs get_current_board_data()")

        time.sleep(0.1)  # Let data accumulate

        # Test get_board_data()
        start_time = time.time()
        data_all = board.get_board_data()
        time1 = time.time() - start_time

        time.sleep(0.05)  # Let more data accumulate

        # Test get_current_board_data()
        start_time = time.time()
        data_current = board.get_current_board_data(50)
        time2 = time.time() - start_time

        print(f"  📊 get_board_data(): {data_all.shape[1]} samples in {time1*1000:.1f}ms")
        if data_all.shape[1] > 0:
            samples_all = data_all[package_num_channel, :]
            print(f"      Range: {samples_all.min():.0f} to {samples_all.max():.0f}")

        print(f"  📊 get_current_board_data(50): {data_current.shape[1]} samples in {time2*1000:.1f}ms")
        if data_current.shape[1] > 0:
            samples_current = data_current[package_num_channel, :]
            print(f"      Range: {samples_current.min():.0f} to {samples_current.max():.0f}")

        print(f"\n🔍 EXPERIMENT 4: Simulate real-time processing timing")

        print("Testing if our processing is too slow...")

        for cycle in range(3):
            cycle_start = time.time()

            # Get data
            data_2d = board.get_board_data()
            data_time = time.time() - cycle_start

            # Simulate processing time
            processing_start = time.time()
            if data_2d.shape[1] > 0:
                # Simulate EEG processing (array operations)
                eeg_data = data_2d[1:17, :].T  # Extract EEG channels
                filtered_data = np.mean(eeg_data, axis=1)  # Simulate filtering
                result = np.sum(filtered_data)  # Simulate computation
            processing_time = time.time() - processing_start

            total_time = time.time() - cycle_start

            if data_2d.shape[1] > 0:
                sample_numbers = data_2d[package_num_channel, :]
                print(f"  Cycle {cycle+1}: {data_2d.shape[1]} samples, range {sample_numbers.min():.0f}-{sample_numbers.max():.0f}")
            else:
                print(f"  Cycle {cycle+1}: No data")

            print(f"    Data fetch: {data_time*1000:.1f}ms, Processing: {processing_time*1000:.1f}ms, Total: {total_time*1000:.1f}ms")

            # Sleep for remainder of 50ms cycle
            remaining_time = 0.05 - total_time
            if remaining_time > 0:
                time.sleep(remaining_time)
                print(f"    Slept: {remaining_time*1000:.1f}ms")
            else:
                print(f"    ⚠️  OVERRUN by {-remaining_time*1000:.1f}ms!")

        board.stop_stream()
        board.release_session()

        print(f"\n{'='*60}")
        print("🔍 DIAGNOSIS COMPLETE")
        print("📋 Key Questions Answered:")
        print("  1. Does get_board_data() give incremental or cumulative data?")
        print("  2. How much data accumulates in different time periods?")
        print("  3. Are we processing too slowly?")
        print("  4. Is the synthetic board behaving realistically?")
        print("="*60)

    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")

if __name__ == "__main__":
    diagnose_timing()
