import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing import (
    load_dataset,
    complex_to_iq,
    normalize_signals,
    create_two_views
)

from contrastive_model import (
    ContrastiveModel,
    nt_xent_loss
)


# ============================================
# Configuration
# ============================================

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

EMBEDDING_DIM = 128
PROJECTION_DIM = 64

MODEL_PATH = "data/contrastive_model.pth"


# ============================================
# Dataset Class
# ============================================

class ContrastiveDataset(Dataset):

    def __init__(self, signals):

        self.signals = signals

    def __len__(self):

        return len(self.signals)

    def __getitem__(self, index):

        signal = self.signals[index]

        # Create two different augmented
        # versions of the same signal
        view_1, view_2 = create_two_views(
            signal
        )

        view_1 = torch.tensor(
            view_1,
            dtype=torch.float32
        )

        view_2 = torch.tensor(
            view_2,
            dtype=torch.float32
        )

        return view_1, view_2


# ============================================
# Main Training Function
# ============================================

def train():

    print("==========================================")
    print("     SELF-SUPERVISED CONTRASTIVE TRAINING")
    print("==========================================")

    # ----------------------------------------
    # 1. Load dataset
    # ----------------------------------------

    print("\nLoading dataset...")

    signals, labels, velocities = (
        load_dataset()
    )

    print("Original signals shape:")
    print(signals.shape)

    # ----------------------------------------
    # 2. Convert complex signals to I/Q
    # ----------------------------------------

    print("\nConverting complex signals to I/Q...")

    iq_signals = complex_to_iq(
        signals
    )

    print("I/Q shape:")
    print(iq_signals.shape)

    # ----------------------------------------
    # 3. Normalize signals
    # ----------------------------------------

    print("\nNormalizing signals...")

    normalized_signals = normalize_signals(
        iq_signals
    )

    print("Normalized shape:")
    print(normalized_signals.shape)

    # ----------------------------------------
    # 4. Create Dataset
    # ----------------------------------------

    dataset = ContrastiveDataset(
        normalized_signals
    )

    # ----------------------------------------
    # 5. Create DataLoader
    # ----------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("\nNumber of training samples:")
    print(len(dataset))

    print("\nBatch size:")
    print(BATCH_SIZE)

    print("\nNumber of batches:")
    print(len(dataloader))

    # ----------------------------------------
    # 6. Select device
    # ----------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nTraining device:")
    print(device)

    # ----------------------------------------
    # 7. Create model
    # ----------------------------------------

    model = ContrastiveModel(
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM
    )

    model = model.to(device)

    # ----------------------------------------
    # 8. Create optimizer
    # ----------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ----------------------------------------
    # 9. Training loop
    # ----------------------------------------

    print("\nStarting training...\n")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for view_1, view_2 in dataloader:

            view_1 = view_1.to(device)
            view_2 = view_2.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # --------------------------------
            # Forward pass
            # --------------------------------

            _, projection_1 = model(
                view_1
            )

            _, projection_2 = model(
                view_2
            )

            # --------------------------------
            # Contrastive loss
            # --------------------------------

            loss = nt_xent_loss(
                projection_1,
                projection_2
            )

            # --------------------------------
            # Backpropagation
            # --------------------------------

            loss.backward()

            # Update model parameters
            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(dataloader)
        )

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Loss: {average_loss:.4f}"
        )

    # ----------------------------------------
    # 10. Save trained model
    # ----------------------------------------

    os.makedirs(
        "data",
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\n==========================================")
    print("       TRAINING COMPLETE")
    print("==========================================")

    print("\nModel saved to:")
    print(MODEL_PATH)


# ============================================
# Run Training
# ============================================

if __name__ == "__main__":

    train()