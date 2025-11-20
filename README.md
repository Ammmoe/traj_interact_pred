# Drone Interaction Classification (Dual-Encoder Bi-GRU)

This repository provides a **training pipeline** for classifying drone interaction types using a **dual-encoder Bi-GRU architecture**. The system processes multi-agent trajectory data and predicts interaction labels between agent pairs.

---

## 📂 Directory Structure

```bash
.
├── train.py                     # Main training script
├── models/
│   ├── bi_gru_encoder.py        # Trajectory embedding encoder
│   └── dual_encoder_classifier.py# Dual-encoder interaction classifier
├── data/
│   ├── data_loader.py           # Dataset loader & preprocessing
│   └── collate_fn.py
├── utils/
│   ├── train_utils.py           # Training, evaluation, scoring helpers
│   └── logger.py                # Logger with timestamped experiment folders
├── experiments/                 # Automatically generated experiment folders
│   ├── 20251119_184413/         # Example experiment folder
│   │   ├── checkpoint.pt        # Saved checkpoint (supports resume training)
│   │   ├── best_model.pt        # Best model based on validation metrics
│   │   ├── last_model.pt        # Final model after last epoch
│   │   ├── config.json          # Saved configurations
│   │   └── training.log         # Detailed train/validation logs
├── requirements.txt
└── README.md
```

---

## 🧱 Environment Setup

It is recommended to use a conda environment.

```bash
# Create and activate the environment
conda create -n drone_interact python=3.10
conda activate drone_interact

# Install dependencies
pip install -r requirements.txt
```

---

## 🔁 Training (`train.py`)

The training pipeline:

1. Loads trajectory and relationship datasets from CSV files
2. Generates agent-pair samples
3. Extracts embeddings using a **Bi-GRU encoder** for each agent
4. Feeds the embeddings into a **dual-encoder classifier**
5. Trains with validation monitoring and optional early stopping
6. Saves:

   * `best_model.pt`
   * `last_model.pt`
   * periodic `checkpoint.pt`
   * `config.json`
   * `training.log`
7. Performs final test-set evaluation

---

## ⚙️ Configuration

Key parameters inside `train.py`:

```python
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
MAX_AGENTS = 6
LOOKBACK = 50
```

Encoder parameters:

```python
encoder_params = {
    "input_size": 6,
    "enc_hidden_size": 64,
    "embedding_dim": 64,
    "num_layers": 1,
}
```

Device selection:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## ▶️ Running Training

To start a new training run:

```bash
python train.py
```

A new folder is automatically created inside `experiments/`:

```bash
experiments/20251120_103050/
```

This folder contains all saved models, logs, and configuration files for that run.

---

## 🔁 Resuming Training

To resume training from a previous experiment checkpoint, set in `train.py`:

```python
RESUME_TRAINING = True
exp_dir = "experiments/20251119_184413"
RESUME_CHECKPOINT = os.path.join(exp_dir, "checkpoint.pt")
```

This restores:

* Model weights
* Optimizer state
* Epoch number
* Training progress

Training continues automatically from the saved checkpoint.

---

## 📊 Evaluation

After training completes, the script:

* Evaluates on the test set
* Computes classification metrics (accuracy, precision, recall, F1)
* Logs results to `training.log`
* Stores metrics in `config.json`

All outputs are saved inside the experiment folder.
