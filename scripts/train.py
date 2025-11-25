"""
Training script for a dual-encoder Bi-GRU model to classify drone interactions.

Features:
- Loads and preprocesses datasets from CSV files.
- Initializes PyTorch models, data loaders, optimizer, and loss function.
- Supports configurable training with validation and early stopping.
- Saves the best model, last model, and periodic checkpoints for pause/resume.
- Allows resuming training from saved checkpoints, preserving optimizer state.
- Logs detailed training, validation, and test metrics, timing, and configuration.
- Performs final evaluation on the test set with metrics and inference timing.
- Saves experiment configuration for reproducibility.
- Uses deterministic seeding optionally for reproducibility.
"""

import os
import time
import json
import torch
from torch.utils.data import DataLoader
from traj_interact_predict.data.data_loader import load_datasets, collate_fn
from traj_interact_predict.models.bi_gru_encoder import TrajEmbeddingExtractor
from traj_interact_predict.models.dual_encoder_classifier import DualEncoderModel
from traj_interact_predict.utils.train_utils import (
    train_one_epoch,
    evaluate_model,
    calculate_evaluation_scores,
)
from traj_interact_predict.utils.logger import get_logger


def main():
    # pylint: disable=all
    RESUME_TRAINING = False
    if RESUME_TRAINING:
        exp_dir = "experiments/20251119_184413"
        RESUME_CHECKPOINT = os.path.join(exp_dir, "checkpoint.pt")
        logger, _ = get_logger(exp_dir=exp_dir)
    else:
        # Setup logger and experiment folder
        logger, exp_dir = get_logger()
        os.makedirs(exp_dir, exist_ok=True)

    # Set deterministic seed for reproducibility
    SET_SEED = False
    if SET_SEED:
        SEED = 42
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
    else:
        SEED = None

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
        "embedding_dim": 64,
        "num_layers": 1,
    }
    dual_encoder_embed_dim = encoder_params["embedding_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    logger.info("\n" + "="*80 + "\nStarting New Experiment\n" + "="*80)
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
    logger.info("Model architecture:\n%s", model)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    train_start_time = time.time()  # Training start time
    best_model_path = os.path.join(exp_dir, "best_model.pt")
    last_model_path = os.path.join(exp_dir, "last_model.pt")
    checkpoint_path = os.path.join(exp_dir, "checkpoint.pt")

    # Early stopping parameters
    start_epoch = 0
    best_val_acc = 0.0
    patience = 5
    epochs_no_improve = 0
    early_stop = False

    # Resume logic
    if RESUME_TRAINING and os.path.exists(RESUME_CHECKPOINT):  # type: ignore
        logger.info("Found checkpoint. Attempting to resume training...")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)  # type: ignore
        try:
            # Load checkpoint
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            best_val_acc = checkpoint.get("best_val_acc", best_val_acc)
            epochs_no_improve = checkpoint.get("epochs_no_improve", epochs_no_improve)
            start_epoch = checkpoint.get("epoch", -1) + 1
            logger.info("Resumed from checkpoint at epoch %d", start_epoch)
        except KeyError as e:
            # If checkpoint keys are missing
            logger.warning(
                "Checkpoint is missing expected keys: %s. Starting from scratch.", e
            )
    else:
        if RESUME_TRAINING:
            # If checkpoint does not exist
            logger.info(
                "Resume requested but no checkpoint found. Starting training from scratch."
            )
        else:
            # Start training from scratch
            logger.info("Starting training from scratch.")

    try:
        for epoch in range(start_epoch, EPOCHS):
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
            if accuracy > best_val_acc:
                best_val_acc = accuracy
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1

            # Save checkpoint each epoch (enables pause/resume)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "epochs_no_improve": epochs_no_improve,
                },
                checkpoint_path,
            )
            logger.info("Checkpoint saved at epoch %d", epoch + 1)

            # Trigger early stopping if no improvements in patience epochs
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered after %d epochs", epoch + 1)
                early_stop = True
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user! Running evaluation...")

    # Save last-epoch model
    finally:
        torch.save(model.state_dict(), last_model_path)

    # If training completed without early stopping
    if not early_stop:
        logger.info("Training finished without early stopping.")

    # Log training time
    train_end_time = time.time()
    training_time = train_end_time - train_start_time
    logger.info("Total training time: %.2f seconds", training_time)

    # Final evaluation on test set
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("Loaded best model for evaluation.")
    else:
        model.load_state_dict(torch.load(last_model_path, map_location=device))
        logger.info("Loaded last model for evaluation (best model not found).")

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
    calculate_evaluation_scores(labels, preds, confidences, "Test", logger, exp_dir)

    config = {
        "device": str(device),
        "model_module": model.__class__.__module__,
        "model_class": model.__class__.__name__,
        "encoder_module": encoder_friendly.__class__.__module__,
        "encoder_class": encoder_friendly.__class__.__name__,
        "encoder_params": encoder_params,
        "MAX_AGENTS": MAX_AGENTS,
        "LOOK_BACK": LOOKBACK,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LR,
        "SET_SEED": SET_SEED,
        "SEED": SEED,
    }

    os.makedirs(exp_dir, exist_ok=True)
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    logger.info("Config saved to %s", config_path)
    
if __name__ == "__main__":
    main()
