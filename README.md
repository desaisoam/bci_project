# BCI Project

A real-time brain-computer interface system that decodes intended movement from EEG signals to control a 2D cursor. Combines a convolutional neural network (EEGNet) for neural feature extraction, an adaptive Kalman filter for smooth state estimation with closed-loop decoder adaptation (CLDA), and an optional AI copilot trained via reinforcement learning.

Built on the RASPy real-time framework with OpenBCI CytonDaisy hardware (16-channel EEG at 125 Hz).

## System Architecture

```
EEG Hardware ──→ Acquisition ──→ Filtering ──→ CNN Decoder ──→ Kalman Filter ──→ Cursor Control
 (OpenBCI)      (BrainFlow)     (IIR SOS)     (EEGNet)        (CLDA)           (PyGame)
  125 Hz         Bipartite      60Hz notch     16D hidden       7D state          ↕
  16 ch          Buffer         4-40Hz BP      + 4 logits       adaptive       AI Copilot
                                                                                (LSTM PPO)
```

### Pipeline

1. **EEG Acquisition** — Raw signals polled from OpenBCI via BrainFlow into a bipartite circular buffer (double-circular structure enabling lock-free contiguous reads)
2. **Filtering** — Cascaded IIR filters (60 Hz notch + 4-40 Hz bandpass) applied sample-by-sample with persistent state
3. **CNN Inference** — EEGNet processes 1-second windows, outputting 4-class direction logits and a 16D hidden feature vector
4. **Kalman Filter** — Estimates cursor velocity from CNN hidden states using a 7D state model `[x, y, v_r, v_l, v_u, v_d, 1]` with online parameter adaptation via sufficient statistics and exponential forgetting
5. **Task & Display** — Center-out cursor control task with hold-to-select, rendered in PyGame
6. **AI Copilot** (optional) — LSTM-based RL agent that blends corrective velocity with the decoder output

## Project Structure

```
├── main/                   # RASPy runtime (process management, shared memory, YAML orchestration)
├── modules/                # Real-time processing modules
│   ├── UpdateEEG*          #   EEG acquisition via BrainFlow
│   ├── filterEEG*          #   IIR filtering (SOS format)
│   ├── decoder_hidden*     #   CNN inference + Welford normalization
│   ├── kf_clda*            #   Kalman filter with closed-loop adaptation
│   ├── kf_util.py          #   KF math (sufficient stats, Sherman-Morrison, steady-state recursion)
│   ├── kf_4_directions*    #   Center-out task state machine
│   ├── buffer_util.py      #   Bipartite circular buffer implementation
│   ├── logger*             #   TCP streaming + binary disk logging
│   └── timer*              #   Master clock (hybrid sleep/busy-wait, ~0.1ms resolution)
├── Offline_EEGNet/         # Offline training pipeline
│   ├── EEGNet.py           #   CNN architecture
│   ├── pipeline_kf_func.py #   End-to-end CNN + KF training
│   ├── train.py            #   Training loop (OneHotMSE loss, 5-fold CV)
│   ├── train_kalmanfilter.py # KF sufficient statistics fitting
│   └── shared_utils/       #   Preprocessing, dataset creation, data loading
├── SJtools/                # AI copilot
│   └── copilot/
│       ├── env.py          #   Gym environment (synthetic decoder simulation)
│       ├── train.py        #   RecurrentPPO training
│       └── copilotUtils/   #   Reward functions, action types (VXY, chargeTargets)
├── models/                 # YAML experiment configurations
│   ├── exp/                #   Production configs
│   └── templates/          #   Config templates
├── experiments/            # Session scripting and protocol helpers
├── scripts/                # CLI utilities (run, validate, analyze)
└── data/raspy/             # Recorded sessions, trained models, KF checkpoints
```

## Getting Started

### Requirements

- Python 3.10+
- CUDA GPU recommended for training
- OpenBCI CytonDaisy + USB dongle (for live sessions)

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU-specific PyTorch builds, install PyTorch first per [pytorch.org](https://pytorch.org/get-started/locally/), then install remaining dependencies.

### Quick Demo (no hardware needed)

```bash
python scripts/run_demo.py
```

Runs a headless playback session using recorded EEG data — filters and logs to disk without any visual display. Useful for verifying the pipeline works end-to-end without hardware.

### Live Session

```bash
python scripts/run_model.py --model exp/open_loop_text_cyton \
    --module_args "/UpdateEEG --serial_port /dev/cu.usbserial-XXXXX" \
    --save True --logfile True
```

## Training Pipeline

### 1. Collect open-loop data

Run the text prompt task to collect labeled EEG data:

```bash
python scripts/run_model.py --model exp/open_loop_text_cyton \
    --module_args "/UpdateEEG --serial_port /dev/cu.usbserial-XXXXX" \
    --data_folder OL_{date}_{counter} --save True
```

### 2. Train CNN + Kalman filter

```bash
cd Offline_EEGNet
python pipeline_kf_func.py config.yaml      # CPU
python pipeline_kf_func.py config_gpu.yaml   # GPU
```

This runs the full pipeline: preprocess EEG, train EEGNet with 5-fold cross-validation (OneHotMSE loss, Adam optimizer), extract hidden states, and fit KF sufficient statistics. Outputs saved to `data/raspy/trained_models/`.

### 3. Train AI copilot (optional)

```bash
python -m SJtools.copilot.train \
    -model=RecurrentPPO -batch_size=512 \
    -action chargeTargets -action_param temperature 1 \
    -obs targetEnd -softmax_type=normal_target \
    -center_out_back -timesteps=2048
```

### 4. Run closed-loop experiment

Reference trained models in a YAML config and run:

```bash
python scripts/run_model.py --model exp/kf-8-directions-gaze \
    --module_args "/UpdateEEG --serial_port /dev/cu.usbserial-XXXXX" \
    --save True --logfile True
```

## Key Technical Details

### Kalman Filter State Model

```
State:  [x_pos, y_pos, v_right, v_left, v_up, v_down, 1]  (7D)
Obs:    CNN hidden state features                           (16D)

Dynamics:    x(t+1) = A @ x(t) + process_noise
Observation: z(t)   = C @ x(t) + obs_noise
Decoding:    x(t+1) = M1 @ x(t) + M2 @ z(t)    (steady-state form)
```

- **A**: Hand-designed from physics (position integrates velocity, velocity decays at 0.825^5 per second)
- **C**: Learned from data via `C = S @ pinv(R)`, with `C[:, 0:2] = 0` (pure velocity filter)
- **CLDA**: Online adaptation using exponentially-weighted sufficient statistics (R, S, T) with Sherman-Morrison incremental matrix updates

### EEGNet Architecture

```
Input: (1, 16 channels, 125 time samples)
  → Temporal Conv(1→8, kernel=51) → BN → DepthwiseConv(8→16, kernel=16) → BN → ELU → Pool → Drop
  → SeparableConv(16→16, kernel=16) → BN → ELU → Pool → Drop
  → Flatten (16D hidden state) → Dense(16→4, max_norm=0.25)
Output: 4-class logits (R, L, U, D) + 16D hidden features for KF
```

### RASPy Framework

Experiments are defined as YAML configs specifying modules (processing nodes) and signals (shared-memory numpy arrays). The runtime spawns each module as a separate process, synchronizing execution via pipe-based DAG scheduling from a master timer. Modules follow a constructor/loop/destructor lifecycle.

## Session Outputs

Each run produces a timestamped directory under `data/raspy/` containing:

| File | Contents |
|------|----------|
| `task.bin` | Decoded positions, targets, game states, decoder outputs |
| `eeg.bin` | Raw + filtered EEG buffers with sample counters |
| `gaze.bin` | Eye tracker data (if enabled) |
| `logfile.log` | Module print output |
| `init_kf.npz` / `final_kf.npz` | KF parameters at session start/end |

## Analysis

```bash
python scripts/analyze_session.py --latest
```

Prints per-stream data rates, label distributions, and counter continuity checks.
