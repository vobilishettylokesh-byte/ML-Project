import numpy as np
import torch

from otfs import create_otfs_grid
from otfs_modulation import otfs_modulate
from channel import wireless_channel

from config import (
    CARRIER_FREQUENCY,
    SNR_DB
)

from preprocessing import (
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

SAMPLING_RATE = 10000

CLASS_NAMES = [
    "Low Doppler",
    "Medium Doppler",
    "High Doppler"
]


# ============================================
# Doppler classification function
# ============================================

def get_expected_class(velocity):

    if velocity < 20:

        return "Low Doppler"

    elif velocity < 80:

        return "Medium Doppler"

    else:

        return "High Doppler"


# ============================================
# Generate new OTFS signal
# ============================================

def generate_signal(velocity):

    # Create delay-Doppler grid
    delay_doppler_grid = (
        create_otfs_grid()
    )

    # OTFS modulation
    _, transmitted_signal = (
        otfs_modulate(
            delay_doppler_grid
        )
    )

    # Wireless channel
    received_signal = wireless_channel(
        transmitted_signal,
        velocity=velocity,
        sampling_rate=SAMPLING_RATE,
        carrier_frequency=CARRIER_FREQUENCY,
        snr_db=SNR_DB
    )

    return received_signal


# ============================================
# Main
# ============================================

def main():

    print("==========================================")
    print("       OTFS DOPPLER PREDICTION")
    print("==========================================")

    # ----------------------------------------
    # 1. Select a test velocity
    # ----------------------------------------

    test_velocity = 50

    print("\nTest velocity:")
    print(test_velocity, "m/s")

    expected_class = get_expected_class(
        test_velocity
    )

    print("\nExpected Doppler regime:")
    print(expected_class)

    # ----------------------------------------
    # 2. Generate signal
    # ----------------------------------------

    print("\nGenerating OTFS signal...")

    received_signal = generate_signal(
        test_velocity
    )

    print("Received signal shape:")
    print(received_signal.shape)

    # ----------------------------------------
    # 3. Convert to I/Q
    # ----------------------------------------

    iq_signal = complex_to_iq(
        received_signal.reshape(1, -1)
    )

    # ----------------------------------------
    # 4. Normalize
    # ----------------------------------------

    normalized_signal = (
        normalize_signals(
            iq_signal
        )
    )

    # ----------------------------------------
    # 5. Convert to PyTorch tensor
    # ----------------------------------------

    signal_tensor = torch.tensor(
        normalized_signal,
        dtype=torch.float32
    )

    # ----------------------------------------
    # 6. Select device
    # ----------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:")
    print(device)

    # ----------------------------------------
    # 7. Load trained encoder
    # ----------------------------------------

    contrastive_model = (
        ContrastiveModel(
            embedding_dim=128,
            projection_dim=64
        )
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

    # ----------------------------------------
    # 8. Extract embedding
    # ----------------------------------------

    signal_tensor = signal_tensor.to(
        device
    )

    with torch.no_grad():

        embedding = encoder(
            signal_tensor
        )

    print("\nEmbedding shape:")
    print(embedding.shape)

    # ----------------------------------------
    # 9. Load classifier
    # ----------------------------------------

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

    # ----------------------------------------
    # 10. Predict
    # ----------------------------------------

    with torch.no_grad():

        output = classifier(
            embedding
        )

        predicted_class = torch.argmax(
            output,
            dim=1
        ).item()

    predicted_name = CLASS_NAMES[
        predicted_class
    ]

    # ----------------------------------------
    # 11. Display result
    # ----------------------------------------

    print("\n==========================================")
    print("              FINAL RESULT")
    print("==========================================")

    print("\nVelocity:")
    print(
        f"{test_velocity} m/s"
    )

    print("\nExpected:")
    print(
        expected_class
    )

    print("\nPredicted:")
    print(
        predicted_name
    )

    print("\n==========================================")

    if predicted_name == expected_class:

        print("Prediction matches expected regime.")

    else:

        print(
            "Prediction differs from the expected "
            "simulation regime."
        )

    print("==========================================")


if __name__ == "__main__":

    main()