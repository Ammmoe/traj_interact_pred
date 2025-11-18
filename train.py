import torch
from torch.utils.data import DataLoader
from data.data_loader import load_datasets, collate_fn
from models.bi_gru_encoder import TrajEmbeddingExtractor
from models.dual_encoder_classifier import DualEncoderModel
from utils.train_utils import train_one_epoch


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
dual_encoder_embed_dim = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_set, val_set, test_set = load_datasets(
    trajectory_csv="data/states.csv",
    relation_csv="data/relations.csv",
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
    val_acc, val_cm = evaluate_model(model, val_loader, device)
    
