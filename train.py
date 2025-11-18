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


# Configuration
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model parameters
encoder_params = {
    "input_size": 6,  # feature dimension
    "enc_hidden_size": 64,
    "dec_hidden_size": 64,
    "num_layers": 1,
}
dual_encoder_embed_dim = encoder_params["dec_hidden_size"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_set, val_set, test_set = load_datasets(
    trajectory_csv="data/drone_states.csv",
    relation_csv="data/drone_relations.csv",
    val_split=VAL_SPLIT,
    test_split=TEST_SPLIT,
    lookback=50,
    device=device,
    max_agents=6,
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

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()

# Training loop
best_val_acc = 0.0
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    logits, preds, labels = evaluate_model(model, val_loader, device)

    # Print epoch results
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")

    # Calculate confidence scores from logits
    confidences = torch.sigmoid(logits).numpy()

    # Calculate and print evaluation scores
    accuracy = calculate_evaluation_scores(
        labels, preds, confidences, f"Epoch {epoch + 1}"
    )

    # Save best model
    if accuracy and accuracy > best_val_acc:
        best_val_acc = accuracy
        torch.save(model.state_dict(), "best_model.pt")

# Final evaluation on test set
model.load_state_dict(torch.load("best_model.pt"))
logits, preds, labels = evaluate_model(model, test_loader, device)
confidences = torch.sigmoid(logits).numpy()
print("\nFinal Evaluation on Test Set:")
calculate_evaluation_scores(labels, preds, confidences, "Test")
