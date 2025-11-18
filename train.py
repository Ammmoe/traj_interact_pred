"""
Training script for a dual-encoder Bi-GRU model to classify drone interactions.

It loads and preprocesses datasets, initializes models and data loaders,
trains the model with validation, and saves the best checkpoint.

After training, it evaluates on the test set, logging loss, metrics,
confusion matrices, calibration curves, and timing information.

All logs and outputs are saved to a timestamped experiment directory
for reproducibility and analysis.

Dependencies: PyTorch, scikit-learn, matplotlib.
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from data.data_loader import load_datasets, collate_fn
from models.bi_gru_encoder import TrajEmbeddingExtractor
from models.dual_encoder_classifier import DualEncoderModel
from utils.train_utils import (
    train_one_epoch,
    evaluate_model,
    calculate_evaluation_scores,
)
from utils.logger import get_logger


# pylint: disable=all
# Configuration
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
MAX_AGENTS = 6
LOOKBACK = 50

# Model parameters
encoder_params = {
    "input_size": 6,  # feature dimension
    "enc_hidden_size": 64,
    "dec_hidden_size": 64,
    "num_layers": 1,
}
dual_encoder_embed_dim = encoder_params["dec_hidden_size"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logger and experiment folder
logger, exp_dir = get_logger()
os.makedirs(exp_dir, exist_ok=True)

# Load datasets
train_set, val_set, test_set = load_datasets(
    trajectory_csv="data/drone_states.csv",
    relation_csv="data/drone_relations.csv",
    val_split=VAL_SPLIT,
    test_split=TEST_SPLIT,
    lookback=LOOKBACK,
    device=device,
    max_agents=MAX_AGENTS,
)

logger.info("Experiment started using device: %s", device)
logger.info("Experiment folder: %s", exp_dir)
logger.info(
    "Total samples (sliding windows): %d", len(train_set) + len(val_set) + len(test_set)
)
logger.info(
    "Train samples: %d, Val samples: %d, Test samples: %d",
    len(train_set),
    len(val_set),
    len(test_set),
)
logger.info(
    "Batch size: %d, Planned epochs: %d, Learning rate: %s", BATCH_SIZE, EPOCHS, LR
)

# DataLoaders ([B, num_drones, lookback, feat_dim])
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# Initialize trajectory encoders
encoder_friendly = TrajEmbeddingExtractor(**encoder_params).to(device)
encoder_unauth = TrajEmbeddingExtractor(**encoder_params).to(device)

# Initialize dual encoder model
model = DualEncoderModel(
    encoder_friendly=encoder_friendly,
    encoder_unauth=encoder_unauth,
    embedding_dim=dual_encoder_embed_dim,
).to(device)

# Log model info
logger.info(
    "Encoder module (friendly agents): %s", encoder_friendly.__class__.__module__
)
logger.info(
    "Encoder module (unauthorized agents): %s", encoder_unauth.__class__.__module__
)
logger.info("Model module: %s", model.__class__.__module__)
logger.info("Encoder architecture (friendly agents):\n%s", encoder_friendly)
logger.info("Encoder architecture (unauthorized agents):\n%s", encoder_unauth)
logger.info("Model architecture:\n%s", model)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()

# Training loop
train_start_time = time.time()  # Training start time
best_model_path = os.path.join(exp_dir, "best_model.pt")

# Early stopping parameters
best_val_acc = 0.0
patience = 5
epochs_no_improve = 0
early_stop = False

for epoch in range(EPOCHS):
    # Training step
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    logger.info("Epoch %d/%d - Train Loss: %.6f", epoch + 1, EPOCHS, train_loss)

    # Evaluation step
    logits, preds, labels, val_loss = evaluate_model(
        model, val_loader, criterion, device
    )

    # Log validation loss
    logger.info(
        "Epoch %d/%d - Validation Loss: %.6f",
        epoch + 1,
        EPOCHS,
        val_loss,
    )

    # Calculate confidence scores from logits
    confidences = torch.sigmoid(logits).numpy()

    # Calculate and print evaluation scores
    accuracy = calculate_evaluation_scores(
        labels, preds, confidences, f"Epoch {epoch + 1}", logger, exp_dir
    )

    # Save best model
    if accuracy and accuracy > best_val_acc:
        best_val_acc = accuracy
        torch.save(model.state_dict(), best_model_path)

# Log training time
train_end_time = time.time()
training_time = train_end_time - train_start_time
logger.info("Total training time: %.2f seconds", training_time)

# Final evaluation on test set
model.load_state_dict(torch.load(best_model_path))

# Test start time
test_start_time = time.time()

# Test step
logits, preds, labels, test_loss = evaluate_model(model, test_loader, criterion, device)

# Log training loss
logger.info("Test Loss: %.6f", test_loss)

# Log testing time
test_end_time = time.time()
testing_time = test_end_time - test_start_time
logger.info("Total inference time: %.2f seconds", testing_time)
logger.info(
    "Avg inference time per batch: %.2f seconds", testing_time / len(test_loader)
)
logger.info(
    "Avg inference time per sequence: %.2f seconds", testing_time / len(test_set)
)

confidences = torch.sigmoid(logits).numpy()
print("\nFinal Evaluation on Test Set:")
calculate_evaluation_scores(labels, preds, confidences, "Test", logger, exp_dir)
