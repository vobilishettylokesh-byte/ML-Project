import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

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

MODEL_PATH = (
    "data/contrastive_model.pth"
)


# ============================================
# Contrastive Dataset
# ============================================

class ContrastiveDataset(Dataset):

    def __init__(self, signals):

        self.signals = signals

    def __len__(self):

        return len(self.signals)

    def __getitem__(self, index):

        signal = self.signals[index]

        # Create two augmented views
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
# Main
# ============================================

def main():

    print("==========================================")
    print("       TRAINING + VALIDATION")
    print("==========================================")

    # ----------------------------------------
    # 1. Load dataset
    # ----------------------------------------

    print("\nLoading dataset...")

    signals, labels, velocities = (
        load_dataset()
    )

    print("\nOriginal dataset:")
    print(signals.shape)

    # ----------------------------------------
    # 2. Convert complex signal to I/Q
    # ----------------------------------------

    print("\nConverting signals to I/Q...")

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
    # 4. Create dataset split
    # ----------------------------------------

    print("\nCreating dataset split...")

    # First split:
    # 80% training
    # 20% temporary

    X_train, X_temp = train_test_split(
        normalized_signals,
        test_size=0.20,
        random_state=42
    )

    # Second split:
    # 10% validation
    # 10% testing

    X_validation, X_test = train_test_split(
        X_temp,
        test_size=0.50,
        random_state=42
    )

    print("\nDataset split:")

    print(
        "Training:",
        len(X_train)
    )

    print(
        "Validation:",
        len(X_validation)
    )

    print(
        "Testing:",
        len(X_test)
    )

    # ----------------------------------------
    # 5. Create datasets
    # ----------------------------------------

    train_dataset = ContrastiveDataset(
        X_train
    )

    validation_dataset = ContrastiveDataset(
        X_validation
    )

    # ----------------------------------------
    # 6. Create DataLoaders
    # ----------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("\nTraining batches:")
    print(len(train_loader))

    print("\nValidation batches:")
    print(len(validation_loader))

    # ----------------------------------------
    # 7. Select device
    # ----------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nTraining device:")
    print(device)

    # ----------------------------------------
    # 8. Create contrastive model
    # ----------------------------------------

    model = ContrastiveModel(
        embedding_dim=EMBEDDING_DIM,
        projection_dim=PROJECTION_DIM
    )

    model = model.to(device)

    # ----------------------------------------
    # 9. Optimizer
    # ----------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ----------------------------------------
    # 10. Store losses
    # ----------------------------------------

    train_losses = []
    validation_losses = []

    # ----------------------------------------
    # 11. Training
    # ----------------------------------------

    print("\n==========================================")
    print("          MODEL TRAINING")
    print("==========================================")

    print()

    for epoch in range(EPOCHS):

        # ====================================
        # Training
        # ====================================

        model.train()

        train_total_loss = 0.0

        for view_1, view_2 in train_loader:

            view_1 = view_1.to(device)
            view_2 = view_2.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            _, projection_1 = model(
                view_1
            )

            _, projection_2 = model(
                view_2
            )

            # Contrastive loss
            loss = nt_xent_loss(
                projection_1,
                projection_2
            )

            # Backpropagation
            loss.backward()

            # Update model
            optimizer.step()

            train_total_loss += (
                loss.item()
            )

        train_loss = (
            train_total_loss /
            len(train_loader)
        )

        # ====================================
        # Validation
        # ====================================

        model.eval()

        validation_total_loss = 0.0

        with torch.no_grad():

            for view_1, view_2 in validation_loader:

                view_1 = view_1.to(device)
                view_2 = view_2.to(device)

                _, projection_1 = model(
                    view_1
                )

                _, projection_2 = model(
                    view_2
                )

                validation_loss = (
                    nt_xent_loss(
                        projection_1,
                        projection_2
                    )
                )

                validation_total_loss += (
                    validation_loss.item()
                )

        validation_loss = (
            validation_total_loss /
            len(validation_loader)
        )

        # Save losses
        train_losses.append(
            train_loss
        )

        validation_losses.append(
            validation_loss
        )

        # Print progress
        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

    # ----------------------------------------
    # 12. Save model
    # ----------------------------------------

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print("\n==========================================")
    print("       CONTRASTIVE MODEL SAVED")
    print("==========================================")

    print("\nModel path:")
    print(MODEL_PATH)

    # ----------------------------------------
    # 13. Create training curve
    # ----------------------------------------

    print("\nCreating training curve...")

    epochs = range(
        1,
        EPOCHS + 1
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        validation_losses,
        marker="s",
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Contrastive Loss"
    )

    plt.title(
        "Contrastive Learning Training Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "data/training_validation_loss.png",
        dpi=300
    )

    plt.show()

    # ----------------------------------------
    # 14. Final message
    # ----------------------------------------

    print("\n==========================================")
    print("          PART 14 COMPLETE")
    print("==========================================")

    print("\nTraining curve saved to:")

    print(
        "data/training_validation_loss.png"
    )


# ============================================
# Run program
# ============================================

if __name__ == "__main__":

    main()