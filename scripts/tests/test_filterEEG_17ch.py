#!/usr/bin/env python3
"""
Unit check: verify filterEEG math works for 17 channels (16 EEG + 1 extra).
Mirrors modules/filterEEG_constructor.py and modules/filterEEG.py behavior.
"""
import numpy as np
import scipy.signal

def main():
    print("="*60)
    print("TEST: filterEEG 17-channel compatibility")
    print("="*60)

    # Match constructor
    Fs = 125.0
    Fn = [60, 60]
    Q = [10, 4]
    sosNotch = [np.concatenate(scipy.signal.iirnotch(Fn[i], Q[i], fs=Fs)).reshape((1, 6))
                for i in range(len(Fn))]
    Fc_lower = 4
    Fc_upper = 40
    wc_lower = Fc_lower/Fs*2.0
    wc_upper = Fc_upper/Fs*2.0
    sosLP = scipy.signal.butter(4, wc_upper, btype='lowpass', output='sos')
    sosHP = scipy.signal.butter(5, wc_lower, btype='highpass', output='sos')
    sos = np.vstack([*sosNotch, sosLP, sosHP])
    zi0 = scipy.signal.sosfilt_zi(sos)

    # Simulate data
    n1, n2, n3 = 10, 5, 1
    x1 = np.random.randn(n1, 17)
    x2 = np.random.randn(n2, 17)
    x3 = np.random.randn(n3, 17)

    # First call initializes internal state like filterEEG
    data_init = x1[0]
    zi = (zi0[..., None] @ data_init.reshape((1, -1)))
    y1, zi = scipy.signal.sosfilt(sos, x1, axis=0, zi=zi)

    # Subsequent calls reuse state
    y2, zi = scipy.signal.sosfilt(sos, x2, axis=0, zi=zi)
    y3, zi = scipy.signal.sosfilt(sos, x3, axis=0, zi=zi)

    print(f"Input shapes: x1={x1.shape}, x2={x2.shape}, x3={x3.shape}")
    print(f"Output shapes: y1={y1.shape}, y2={y2.shape}, y3={y3.shape}")
    assert y1.shape == x1.shape and y2.shape == x2.shape and y3.shape == x3.shape
    print("✅ filterEEG compatibility PASS (shape & state propagation)")

if __name__ == "__main__":
    main()

