# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### CNN-KF Training
Train the convolutional neural network and Kalman filter:
```bash
cd Offline_EEGNet
python pipeline_kf_func.py config.yaml          # CPU training
python pipeline_kf_func.py config_gpu.yaml      # GPU training
```

### Copilot Training
Train the LSTM copilot:
```bash
python -m SJtools.copilot.train -model=RecurrentPPO -batch_size=512 -action chargeTargets -action_param temperature 1 -obs targetEnd -holdtime=2.0 -stillCS=0.0 -lr_scheduler=constant -softmax_type=normal_target -velReplaceSoftmax -no_wandb -reward_type=baseLinDist.yaml -center_out_back -extra_targets_yaml=dir-8.yaml -timesteps=2048 -n_steps=512 -renderTrain
```

### Running RASPy Models
```bash
python ./main/main.py $model_name    # Replace $model_name with relative path from models/ directory
```

### Demo Commands
```bash
python main/main2b.py replay_demo              # Replay center-out task demo
python main/main2b.py exp/SJ-text-gaze_demo    # Text experiment demo (press escape to quit)
python -m SJtools.copilot.test models/keep/charge/T8B_LSTM2_truedecay/best_model -center_out_back -softmax_type=normal_target
```

### Python Environment
Python 3.8 required. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: PyTorch installation may require device-specific setup. Some packages require specific versions: `setuptools==65.5.0` and `wheel==0.38.4`.

## Project Architecture

### Core Components

**RASPy Framework**: Pure Python real-time experimental framework based on LiCoRICE. Uses modular architecture with:
- **Modules**: Define executable code with constructor/loop/destructor lifecycle
- **Signals**: Data transfer between modules
- **Models**: YAML configuration files defining module relationships and timing

**CNN-KF System**: Convolutional neural network combined with Kalman filter for brain-machine interface control that adapts decoder parameters in closed loop.

**AI Copilot**: LSTM-based system that modifies action distributions based on environmental observations and task structure.

### Directory Structure

- `Offline_EEGNet/`: Offline CNN training pipeline and utilities
- `main/`: Core RASPy execution scripts and utilities
- `modules/`: RASPy modules for real-time data processing (EEG, filtering, logging, etc.)
- `decoders/`: Neural network decoder implementations
- `models/`: RASPy model configuration files (.yaml)
- `SJtools/`: Copilot training and testing utilities
- `stream/`: EEG data streaming code
- `data/`: Experimental data and trained models

### Key Modules

**Standard RASPy Modules**:
- `timer`: Maintains real-world clock synchronicity (~0.1ms resolution)
- `UpdateEEG`: Receives EEG samples and stores in circular buffer
- `filterEEG`: Linear filtering of EEG data in SOS format
- `decoder`: Imports and runs neural network decoders
- `logger`/`logger_disk`: Data logging and disk saving
- `task_module`: Task variable manipulation and state management

**Specialized Modules**:
- `kf_clda`: Kalman filter with closed-loop decoder adaptation (CLDA)
- `recv_gaze_buffer`: Tobii eye tracker data reception
- `SJ_text_classification`: Text classification for BCI spelling tasks

### Data Processing Pipeline

1. **EEG Acquisition**: 1000Hz sampling, 64+3 channels via ANT Neuro eego rt
2. **Preprocessing**: Linear filtering and circular buffer management
3. **Neural Decoding**: CNN feature extraction → velocity estimates
4. **Kalman Filtering**: State estimation with adaptive parameters
5. **Copilot Integration**: Action distribution modification based on task context
6. **Real-time Feedback**: PyGame-based visual feedback system

### Configuration Management

RASPy models use YAML configuration with:
- Module definitions (constructor/loop/destructor flags)
- Signal routing and synchronization dependencies
- Parameter specification via `params` field
- Group parameters for applying settings to multiple modules

### Important Notes

- Working directory changes to repository root when running RASPy models
- Real-time constraints: modules cannot use threading/multiprocessing during loops
- GPU acceleration available for neural network training and inference
- Gaze data normalized to monitor resolution (Tobii Pro Nano @ 60Hz)
- Set `quit_=True` in any module to signal clean shutdown