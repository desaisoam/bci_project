
# get data from buffer
idx = eegbufferindex[0]
window = np.copy(databuffer[idx - input_length:idx])

# Remove non-EEG columns (e.g., sample counter) before preprocessing/resampling
# Assumes the counter is stored in the last column as written by UpdateEEG/filterEEG
if window.shape[1] > 0:
    data = window[:, :-1]
else:
    data = window

# preprocess data
data = dataPreprocessor.preprocess(data)

# downsample
data = resample(data, downsampled_length, axis=0)

# predict using model
with torch.no_grad():
    data = torch.FloatTensor(data).to(device)
    probs, logits, hidden_state = model(data, return_logits=False, return_dataclass=True) # misnomer. returns (softmax_probs, logits, hidden_state)
    probs, logits, hidden_state = probs.flatten(), logits.flatten(), hidden_state.flatten()
    # Warning: this stores logits, not softmax probabilities!
    decoder_output.fill(0)
    decoder_h.fill(0)
    decoder_output[0:len(probs)] = logits.detach().cpu().numpy()
    decoder_h[0:len(hidden_state)] = hidden_state.detach().cpu().numpy()
