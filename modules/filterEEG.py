new_idx = eegbufferindex[0]

nSamples = (new_idx - idx) % N
if idx != new_idx:
    data = eegbuffersignal[new_idx-nSamples:new_idx]
    if zi is None:
        data_init = data[0]
        zi = (zi0[..., None]@data_init.reshape((1, -1)))
    # May want to filter sample-by-sample if non-linear filtering is used
    # Preserve last column (continuous counter) unfiltered
    counter_col = data[:, -1].copy()
    data_filt, zo = scipy.signal.sosfilt(sos, data, axis=0, zi=zi)
    data_filt[:, -1] = counter_col
    zi = zo
    
    databuffer[new_idx-nSamples:new_idx] = data_filt
    wrap_inds = (np.arange(new_idx-nSamples, new_idx) - N) % (2*N)
    databuffer[wrap_inds] = data_filt
    
idx = new_idx
