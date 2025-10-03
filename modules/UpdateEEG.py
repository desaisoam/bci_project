'''Get EEG data from BrainFlow (live, playback, or synthetic)'''

# Get new data from BrainFlow (replaces TCP receiving)
try:
    # Get all available data and remove from BrainFlow buffer
    data_2d = board.get_board_data()

    if data_2d.shape[1] > 0:
        # BrainFlow returns [channels × samples], we need [samples × channels]
        eeg_data = data_2d[eeg_channels, :].T  # (n_samples, 16)
        nSamples = eeg_data.shape[0]

        # Cache values used to compute synthetic continuous counter
        base_total = totalSamples

        # For real package counter, get raw values as int64 for unwrapping; else, use None
        raw_pkg = None
        if 'has_pkg_channel' in globals() and has_pkg_channel:
            raw_pkg = data_2d[package_num_channel, :].astype(np.int64)

        # Skip initial samples if needed
        # Note: We still write samples but mark nSamples=0 if skipping is active
        skipping = (base_total + nSamples) < samplesToSkip

        for idx in range(nSamples):
            # Compute continuous counter value per sample
            if raw_pkg is None:
                cont = base_total + idx
            else:
                pkg = int(raw_pkg[idx])
                # unwrap across modulus
                if 'last_pkg' in globals() and last_pkg is not None:
                    if pkg < last_pkg and (last_pkg - pkg) > (pkg_modulus // 2):
                        # wrap detected
                        pkg_epoch += pkg_modulus
                last_pkg = pkg
                cont = pkg_epoch + pkg

            # Compose row and write to bipartite buffer
            # Use buffer dtype for consistency
            row = np.empty((nChannels,), dtype=eegbuffersignal.dtype)
            row[:16] = eeg_data[idx]
            row[16] = cont
            eegbuffersignal[bufferInd] = row
            eegbuffersignal[bufferInd - bufferoffset] = row
            bufferInd = (bufferInd - bufferoffset + 1) % bufferoffset + bufferoffset

            # Continuity check on continuous counter
            if prevCount > 0 and (cont != prevCount + 1):
                print(f"!!!!!!!!! Missing data between samples {prevCount} and {cont}")
            prevCount = cont

        totalSamples += nSamples
        if skipping:
            nSamples = 0
    else:
        nSamples = 0

except Exception as e:
    print(f"BrainFlow error: {e}")
    nSamples = 0

numEEGSamples[:] = nSamples
eegbufferindex[:] = bufferInd
# totalValidEEGSamples counts valid samples after any initial skip
totalValidEEGSamples += nSamples

t0 = time.time()
tickNo += 1
