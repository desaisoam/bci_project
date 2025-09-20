'''Get EEG data from OpenBCI via BrainFlow'''

# Get new data from BrainFlow (replaces TCP receiving)
try:
    # Get all available data and remove from BrainFlow buffer (like original TCP approach)
    data_2d = board.get_board_data()  # Get ALL new data since last call, removes from buffer

    if data_2d.shape[1] > 0:  # If we have new samples
        # BrainFlow returns [channels × samples], we need [samples × channels]
        # Extract EEG channels and REAL sample counter
        eeg_data = data_2d[eeg_channels, :].T  # Shape: (n_samples, 16)
        real_sample_counter = data_2d[package_num_channel, :].T.reshape(-1, 1)  # Shape: (n_samples, 1) - REAL counter from hardware
        nSamples = eeg_data.shape[0]

        # Combine EEG data with REAL sample counter (not fake generated one)
        data = np.concatenate([eeg_data, real_sample_counter], axis=1)  # Shape: (n_samples, 17)

        totalSamples += nSamples

        # Skip initial samples if needed (keep existing logic)
        if totalSamples < samplesToSkip:
            nSamples = 0
        else:
            # Process each sample (keep existing buffer logic)
            for idx in range(data.shape[0]):
                # directly modify the signal to save time
                # to access, use eegbuffersignal[eegbufferindex - (number_of_samples-1):eegbufferindex+1]
                eegbuffersignal[bufferInd] = data[idx]  # All 17 columns (16 EEG + timestamp)
                eegbuffersignal[bufferInd - bufferoffset] = data[idx]  # All 17 columns (16 EEG + timestamp)
                bufferInd = (bufferInd - bufferoffset + 1) % bufferoffset + bufferoffset

                # Sample continuity check using REAL hardware sample counter
                real_sample_num = data[idx, -1]  # Real sample counter from hardware
                if real_sample_num != prevCount + 1 and prevCount > 0:
                    print("!!!!!!!!! Missing data between samples {} and {}".format(prevCount, real_sample_num))
                prevCount = real_sample_num
    else:
        nSamples = 0

except Exception as e:
    print(f"BrainFlow error: {e}")
    nSamples = 0

numEEGSamples[:] = nSamples
eegbufferindex[:] = bufferInd
# is NOT the same as totalSamples, but rather the total samples
# after surpassing samplesToSkip
totalValidEEGSamples += nSamples

t0 = time.time()
tickNo += 1
