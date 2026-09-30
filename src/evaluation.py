import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)

from preprocessing import (
    load_dataset,
    complex_to_iq,
    normalize_signals
)

from contrastive_model import ContrastiveModel
from classifier import DopplerClassifier


# ============================================
# Configuration
# ============================================

CONTRASTIVE_MODEL_PATH = (
    "data/contrastive_model.pth"
)

CLASSIFIER_MODEL_PATH = (
    "data/doppler_classifier.pth"
)

BATCH_SIZE = 32

CLASS_NAMES = [
    "Low Doppler",
    "Medium Doppler",
    "High Doppler"
]


# ============================================
# Extract Embeddings
# ============================================

def extract_embeddings(
    encoder,
    data,
    device
):

    data_tensor = torch.tensor(
        data,
        dtype=torch.float32
    )

    embeddings_list = []

    encoder.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(data_tensor),
            BATCH_SIZE
        ):

            batch = data_tensor[
                start:start + BATCH_SIZE
            ]

            batch = batch.to(device)

            embeddings = encoder(
                batch
            )

            embeddings_list.append(
                embeddings.cpu()
            )

    return torch.cat(
        embeddings_list
    )


# ============================================
# Main
# ============================================

def main():

    print("==========================================")
    print("          MODEL EVALUATION")
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

    # ----------------------------------------
    # 2. Convert to I/Q
    # ----------------------------------------

    print("\nConverting signals to I/Q...")

    iq_signals = complex_to_iq(
        signals
    )

    print("I/Q shape:")
    print(iq_signals.shape)

    # ----------------------------------------
    # 3. Normalize
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

    print("\nCreating test set...")

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

    print("Testing samples:")
    print(len(X_test))

    # ----------------------------------------
    # 5. Select device
    # ----------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:")
    print(device)

    # ----------------------------------------
    # 6. Load contrastive model
    # ----------------------------------------

    print("\nLoading contrastive model...")

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

    encoder = contrastive_model.encoder

    encoder.eval()

    print("Contrastive encoder loaded.")

    # ----------------------------------------
    # 7. Extract embeddings
    # ----------------------------------------

    print("\nExtracting training embeddings...")

    train_embeddings = extract_embeddings(
        encoder,
        X_train,
        device
    )

    print(
        "Training embeddings:",
        train_embeddings.shape
    )

    print("\nExtracting testing embeddings...")

    test_embeddings = extract_embeddings(
        encoder,
        X_test,
        device
    )

    print(
        "Testing embeddings:",
        test_embeddings.shape
    )

    # ----------------------------------------
    # 8. Load classifier
    # ----------------------------------------

    print("\nLoading trained classifier...")

    classifier = DopplerClassifier(
        embedding_dim=128,
        num_classes=3
    )

    classifier.load_state_dict(
        torch.load(
            CLASSIFIER_MODEL_PATH,
            map_location=device
        )
    )

    classifier = classifier.to(device)

    classifier.eval()

    print("Classifier loaded successfully.")

    # ----------------------------------------
    # 9. Predict test samples
    # ----------------------------------------

    print("\nGenerating predictions...")

    test_embeddings = (
        test_embeddings.to(device)
    )

    with torch.no_grad():

        outputs = classifier(
            test_embeddings
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

    predictions = predictions.cpu().numpy()

    # ----------------------------------------
    # 10. Calculate accuracy
    # ----------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n==========================================")
    print("              RESULTS")
    print("==========================================")

    print(
        f"\nTest Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # ----------------------------------------
    # 11. Classification report
    # ----------------------------------------

    print("\nClassification Report:\n")

    report = classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES
    )

    print(report)

    # ----------------------------------------
    # 12. Confusion matrix
    # ----------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix:")
    print(cm)

    # ----------------------------------------
    # 13. Display confusion matrix
    # ----------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot()

    plt.title(
        "Doppler Regime Classification"
    )

    plt.tight_layout()

    plt.savefig(
        "data/confusion_matrix.png",
        dpi=300
    )

    plt.show()

    # ----------------------------------------
    # 14. Actual vs Predicted
    # ----------------------------------------

    actual_counts = [
        np.sum(
            y_test == i
        )
        for i in range(3)
    ]

    predicted_counts = [
        np.sum(
            predictions == i
        )
        for i in range(3)
    ]

    x = np.arange(
        len(CLASS_NAMES)
    )

    width = 0.35

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        x - width / 2,
        actual_counts,
        width,
        label="Actual"
    )

    plt.bar(
        x + width / 2,
        predicted_counts,
        width,
        label="Predicted"
    )

    plt.xlabel(
        "Doppler Regime"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "Actual vs Predicted Doppler Classes"
    )

    plt.xticks(
        x,
        CLASS_NAMES
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "data/prediction_distribution.png",
        dpi=300
    )

    plt.show()

    # ----------------------------------------
    # 15. Final message
    # ----------------------------------------

    print("\n==========================================")
    print("        EVALUATION COMPLETE")
    print("==========================================")

    print("\nSaved graphs:")

    print(
        "data/confusion_matrix.png"
    )

    print(
        "data/prediction_distribution.png"
    )


# ============================================
# Run program
# ============================================

if __name__ == "__main__":

    main()