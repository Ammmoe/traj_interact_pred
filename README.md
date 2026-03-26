# Drone Interaction Classification (Dual-Encoder Bi-GRU)

This repository provides a **training pipeline** for classifying drone interaction types using a **dual-encoder Bi-GRU architecture**. The system processes multi-agent trajectory data and predicts interaction labels between agent pairs.

---

## Model Architecture

### Attention-Enhanced Dual-Encoder Classifier

Two Bi-GRU encoders independently generate embeddings for each agent type.  
A Cross-Agent Self-Attention Transformer captures interactions between agent embeddings, followed by an MLP classifier to predict interaction types.

> **Figure 1: Attention-Enhanced Dual-Encoder Classifier**  
> *(High-level overview of the full model)*

![Dual Encoder Architecture](Dual-Encoder-Classifier.png)

---

## Environment Setup (ROS2 + PyTorch via venv)

```bash
# Clone the repository
git clone git@bitbucket.org:nusuav/traj_interact_predict.git
cd traj_interact_predict

# Install system dependencies
sudo apt update
sudo apt install -y python3-venv python3-pip python3-pandas python3-sklearn python3-tqdm python3-matplotlib

# Create and activate venv
python3 -m venv ~/venvs/fasttask --system-site-packages
source ~/venvs/fasttask/bin/activate

# Install PyTorch inside venv
python -m pip install -U pip
python -m pip install torch==2.10.0

# Add to ~/.bashrc
source ~/venvs/fasttask/bin/activate
export ROS2_INSTALL_PATH=/opt/ros/humble
source ${ROS2_INSTALL_PATH}/setup.bash
source ~/hifisim_ws/install/setup.bash

# Close the terminal and open a new terminal or source ~/.bashrc to apply changes

# Verify interpretor
which python3
python3 -c "import sys; print(sys.executable)"
python3 -c "import rclpy; print(rclpy.__file__)"
python3 -c "import torch; print(torch.__version__)"

# Install Git Large File Storage (Git LFS)
git lfs install

# Pull large files tracked by Git LFS (e.g., datasets)
git lfs pull
```

> ⚠️ Do **not** use `sudo pip install` for system Python.

---

## Training

### Run training via ROS2

```bash
ros2 run traj_interact_predict train
```

All outputs (checkpoints, logs, configs) will be saved automatically in `experiments/`.

---

### Training Configuration

Key parameters are set in the training script:

```bash
BATCH_SIZE = 32     # Number of samples per batch  
EPOCHS = 50         # Training iterations over the dataset  
LR = 1e-3           # Optimizer learning rate  
VAL_SPLIT = 0.15    # Validation data split  
TEST_SPLIT = 0.15   # Test data split  
MAX_AGENTS = 6      # Max agents per sample  
LOOKBACK = 50       # Number of past timesteps used  
```

Encoder parameters:

```bash
encoder_params = {  
    "input_size": 6,  
    "enc_hidden_size": 64,  
    "embedding_dim": 64,  
    "num_layers": 1  
}
```

---

### Resuming Training

Set in the training script:

```bash
RESUME_TRAINING = True  
exp_dir = "experiments/<previous_experiment>"  
```

This restores model weights, optimizer state, and epoch number.

---

## Inference

### Run inference via ROS2

```bash
ros2 run traj_interact_predict inference
```

Optional parameters allow simulating missing agents:

- `num_friendly_to_pad` – number of friendly agents to zero-pad  
- `num_unauth_to_pad` – number of unauthorized agents to zero-pad

---

## Simulation Agent

### Run the simulation agent via ROS2

```bash
ros2 run traj_interact_predict sim_agent
```

The simulation agent:

- Initializes agent tracking  
- Generates trajectories for multiple agents  
- Predicts interactions between friendly and unauthorized agents  
- Prints predicted pairs with confidence probabilities 

---
