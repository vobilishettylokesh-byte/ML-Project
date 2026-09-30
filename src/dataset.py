import os
import numpy as np

from otfs import create_otfs_grid
from otfs_modulation import otfs_modulate
from channel import wireless_channel
from config import (
    CARRIER_FREQUENCY,
    SNR_DB
)


# ============================================
# Dataset configuration
# ============================================

NUM_SAMPLES_PER_CLASS = 300

SAMPLING_RATE = 10000


# Doppler velocity ranges
LOW_VELOCITY = (5, 19)

MEDIUM_VELOCITY = (20, 79)

HIGH_VELOCITY = (80, 150)


# ============================================
# Generate one OTFS sample
# ============================================

def generate_one_sample(velocity):
    """
    Generate one OTFS received signal
    for a given velocity.
    """

    # Create delay-Doppler grid
    delay_doppler_grid = create_otfs_grid()

    # OTFS modulation
    _, transmitted_signal = otfs_modulate(
        delay_doppler_grid
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
# Generate complete dataset
# ============================================

def generate_dataset():

    signals = []

    labels = []

    velocities = []

    print("==========================================")
    print("       OTFS DOPPLER DATASET")
    print("==========================================")

    # ----------------------------------------
    # Class 0 - Low Doppler
    # ----------------------------------------

    print("\nGenerating Low Doppler samples...")

    for i in range(NUM_SAMPLES_PER_CLASS):

        velocity = np.random.uniform(
            LOW_VELOCITY[0],
            LOW_VELOCITY[1]
        )

        signal = generate_one_sample(
            velocity
        )

        signals.append(signal)

        labels.append(0)

        velocities.append(velocity)

    print("Low Doppler samples:",
          NUM_SAMPLES_PER_CLASS)


    # ----------------------------------------
    # Class 1 - Medium Doppler
    # ----------------------------------------

    print("\nGenerating Medium Doppler samples...")

    for i in range(NUM_SAMPLES_PER_CLASS):

        velocity = np.random.uniform(
            MEDIUM_VELOCITY[0],
            MEDIUM_VELOCITY[1]
        )

        signal = generate_one_sample(
            velocity
        )

        signals.append(signal)

        labels.append(1)

        velocities.append(velocity)

    print("Medium Doppler samples:",
          NUM_SAMPLES_PER_CLASS)


    # ----------------------------------------
    # Class 2 - High Doppler
    # ----------------------------------------

    print("\nGenerating High Doppler samples...")

    for i in range(NUM_SAMPLES_PER_CLASS):

        velocity = np.random.uniform(
            HIGH_VELOCITY[0],
            HIGH_VELOCITY[1]
        )

        signal = generate_one_sample(
            velocity
        )

        signals.append(signal)

        labels.append(2)

        velocities.append(velocity)

    print("High Doppler samples:",
          NUM_SAMPLES_PER_CLASS)


    # ========================================
    # Convert to NumPy arrays
    # ========================================

    signals = np.array(signals)

    labels = np.array(labels)

    velocities = np.array(velocities)


    # ========================================
    # Create data directory
    # ========================================

    os.makedirs("data", exist_ok=True)


    # ========================================
    # Save dataset
    # ========================================

    np.save(
        "data/signals.npy",
        signals
    )

    np.save(
        "data/labels.npy",
        labels
    )

    np.save(
        "data/velocities.npy",
        velocities
    )


    # ========================================
    # Display information
    # ========================================

    print("\n==========================================")
    print("       DATASET GENERATION COMPLETE")
    print("==========================================")

    print("\nSignals shape:")
    print(signals.shape)

    print("\nLabels shape:")
    print(labels.shape)

    print("\nVelocities shape:")
    print(velocities.shape)

    print("\nClass distribution:")

    print(
        "Low Doppler    :",
        np.sum(labels == 0)
    )

    print(
        "Medium Doppler :",
        np.sum(labels == 1)
    )

    print(
        "High Doppler   :",
        np.sum(labels == 2)
    )

    print("\nDataset saved successfully!")


# ============================================
# Main
# ============================================

if __name__ == "__main__":

    # Reproducibility
    np.random.seed(42)

    generate_dataset()