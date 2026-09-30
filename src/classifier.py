import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import (
    load_dataset,
    complex_to_iq,
    normalize_signals
)

from contrastive_model import ContrastiveModel


# ============================================
# Configuration
# ============================================

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

NUM_CLASSES = 3

CONTRASTIVE_MODEL_PATH = (
    "data/contrastive_model.pth"
)

CLASSIFIER_MODEL_PATH = (
    "data/doppler_classifier.pth"
)


# ============================================
# Doppler Classifier
# ============================================

class DopplerClassifier(nn.Module):
    """
    Classifier for Doppler regime classification.

    Input:
        128-dimensional CNN embedding

    Output:
        3 classes

        0 -> Low Doppler
        1 -> Medium Doppler
        2 -> High Doppler
    """

    def __init__(
        self,
        embedding_dim=128,
        num_classes=3
    ):

        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                embedding_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        return self.classifier(x)


# ============================================
# Extract Embeddings
# ============================================

def extract_embeddings(
    encoder,
    data,
    device
):
    """
    Convert signals into CNN embeddings.
    """

    data_tensor = torch.tensor(
        data,
        dtype=torch.float32
    )

    loader = DataLoader(
        data_tensor,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_embeddings = []

    encoder.eval()

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            embedding = encoder(
                batch
            )

            all_embeddings.append(
                embedding.cpu()
            )

    return torch.cat(
        all_embeddings
    )


# ============================================
# Main
# ============================================

def main():

    print("==========================================")
    print("       DOPPLER REGIME CLASSIFICATION")
    print("==========================================")

    # ----------------------------------------
    # 1. Load dataset
    # ----------------------------------------

    print("\nLoading dataset...")

    signals, labels, velocities = (
        load_dataset()
    )

    print("\nSignals shape:")
    print(signals.shape)

    print("\nLabels shape:")
    print(labels.shape)

    print("\nVelocities shape:")
    print(velocities.shape)

    # ----------------------------------------
    # 2. Convert complex signals to I/Q
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
    # 4. Train/Test split
    # ----------------------------------------

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = (
        train_test_split(
            normalized_signals,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
    )

    print("\nTraining samples:")
    print(len(X_train))

    print("\nTesting samples:")
    print(len(X_test))

    # ----------------------------------------
    # 5. Select device
    # ----------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nTraining device:")
    print(device)

    # ----------------------------------------
    # 6. Load trained contrastive model
    # ----------------------------------------

    print("\nLoading trained contrastive model...")

    contrastive_model = ContrastiveModel(
        embedding_dim=128,
        projection_dim=64
    )

    contrastive_model.load_state_dict(
        torch.load(
            CONTRASTIVE_MODEL_PATH,
            map_location=device
        )
    )

    contrastive_model = (
        contrastive_model.to(device)
    )

    # ----------------------------------------
    # 7. Get CNN encoder
    # ----------------------------------------

    encoder = contrastive_model.encoder

    # Freeze encoder parameters
    for parameter in encoder.parameters():

        parameter.requires_grad = False

    encoder.eval()

    print("Pre-trained encoder loaded.")

    # ----------------------------------------
    # 8. Extract training embeddings
    # ----------------------------------------

    print("\nExtracting training embeddings...")

    train_embeddings = extract_embeddings(
        encoder,
        X_train,
        device
    )

    print("Training embeddings:")
    print(train_embeddings.shape)

    # ----------------------------------------
    # 9. Extract testing embeddings
    # ----------------------------------------

    print("\nExtracting testing embeddings...")

    test_embeddings = extract_embeddings(
        encoder,
        X_test,
        device
    )

    print("Testing embeddings:")
    print(test_embeddings.shape)

    # ----------------------------------------
    # 10. Convert labels to tensors
    # ----------------------------------------

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )

    # ----------------------------------------
    # 11. Create classifier
    # ----------------------------------------

    print("\nCreating Doppler classifier...")

    classifier = DopplerClassifier(
        embedding_dim=128,
        num_classes=NUM_CLASSES
    )

    classifier = classifier.to(device)

    # ----------------------------------------
    # 12. Create training dataset
    # ----------------------------------------

    train_dataset = TensorDataset(
        train_embeddings,
        y_train_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ----------------------------------------
    # 13. Loss function
    # ----------------------------------------

    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------
    # 14. Optimizer
    # ----------------------------------------

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LEARNING_RATE
    )

    # ----------------------------------------
    # 15. Train classifier
    # ----------------------------------------

    print("\n==========================================")
    print("          CLASSIFIER TRAINING")
    print("==========================================")

    print()

    for epoch in range(EPOCHS):

        classifier.train()

        total_loss = 0.0

        for embeddings, targets in train_loader:

            embeddings = embeddings.to(
                device
            )

            targets = targets.to(
                device
            )

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = classifier(
                embeddings
            )

            # Calculate loss
            loss = criterion(
                outputs,
                targets
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss /
            len(train_loader)
        )

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Loss: {average_loss:.4f}"
        )

    # ----------------------------------------
    # 16. Evaluate classifier
    # ----------------------------------------

    print("\n==========================================")
    print("              EVALUATION")
    print("==========================================")

    classifier.eval()

    test_embeddings_device = (
        test_embeddings.to(device)
    )

    with torch.no_grad():

        outputs = classifier(
            test_embeddings_device
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

    predictions = predictions.cpu().numpy()

    # ----------------------------------------
    # 17. Calculate accuracy
    # ----------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nTest Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # ----------------------------------------
    # 18. Classification report
    # ----------------------------------------

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Low Doppler",
                "Medium Doppler",
                "High Doppler"
            ]
        )
    )

    # ----------------------------------------
    # 19. Save classifier
    # ----------------------------------------

    torch.save(
        classifier.state_dict(),
        CLASSIFIER_MODEL_PATH
    )

    print("\n==========================================")
    print("          CLASSIFIER SAVED")
    print("==========================================")

    print("\nSaved model:")
    print(
        CLASSIFIER_MODEL_PATH
    )

    # ----------------------------------------
    # 20. Final message
    # ----------------------------------------

    print("\n==========================================")
    print("       CLASSIFICATION COMPLETE")
    print("==========================================")


# ============================================
# Run Program
# ============================================

if __name__ == "__main__":

    main()