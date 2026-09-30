import numpy as np
import torch


# ============================================
# Load dataset
# ============================================

def load_dataset():

    signals = np.load("data/signals.npy")

    labels = np.load("data/labels.npy")

    velocities = np.load(
        "data/velocities.npy"
    )

    return signals, labels, velocities


# ============================================
# Convert complex signal to I/Q channels
# ============================================

def complex_to_iq(signals):
    """
    Convert complex signals into two channels.

    Channel 0 -> In-phase (I)
    Channel 1 -> Quadrature (Q)

    Input:
        (samples, signal_length)

    Output:
        (samples, 2, signal_length)
    """

    real_part = np.real(signals)

    imaginary_part = np.imag(signals)

    iq = np.stack(
        [real_part, imaginary_part],
        axis=1
    )

    return iq.astype(np.float32)


# ============================================
# Normalize signal
# ============================================

def normalize_signals(signals):
    """
    Normalize each signal independently.

    This prevents signals with larger amplitude
    from dominating the learning process.
    """

    mean = np.mean(
        signals,
        axis=2,
        keepdims=True
    )

    std = np.std(
        signals,
        axis=2,
        keepdims=True
    )

    std = np.maximum(std, 1e-8)

    normalized = (
        signals - mean
    ) / std

    return normalized.astype(np.float32)


# ============================================
# Add Gaussian noise
# ============================================

def add_noise(signal, noise_level=0.02):

    noise = np.random.normal(
        0,
        noise_level,
        size=signal.shape
    )

    return signal + noise.astype(
        np.float32
    )


# ============================================
# Random amplitude scaling
# ============================================

def random_amplitude(signal):

    scale = np.random.uniform(
        0.9,
        1.1
    )

    return signal * scale


# ============================================
# Random time shift
# ============================================

def random_shift(signal, max_shift=10):

    shift = np.random.randint(
        -max_shift,
        max_shift + 1
    )

    return np.roll(
        signal,
        shift,
        axis=-1
    )


# ============================================
# Create augmented view
# ============================================

def create_augmented_view(signal):

    augmented = signal.copy()

    # Random noise
    augmented = add_noise(
        augmented,
        noise_level=0.02
    )

    # Random amplitude
    augmented = random_amplitude(
        augmented
    )

    # Random time shift
    augmented = random_shift(
        augmented,
        max_shift=10
    )

    return augmented.astype(
        np.float32
    )


# ============================================
# Create two contrastive views
# ============================================

def create_two_views(signal):

    view_1 = create_augmented_view(
        signal
    )

    view_2 = create_augmented_view(
        signal
    )

    return view_1, view_2


# ============================================
# Main test
# ============================================

if __name__ == "__main__":

    print("===================================")
    print("   PREPROCESSING TEST")
    print("===================================")

    # Load data
    signals, labels, velocities = (
        load_dataset()
    )

    print("\nOriginal dataset:")
    print(signals.shape)

    # Convert complex -> I/Q
    iq_signals = complex_to_iq(
        signals
    )

    print("\nAfter I/Q conversion:")
    print(iq_signals.shape)

    # Normalize
    normalized = normalize_signals(
        iq_signals
    )

    print("\nAfter normalization:")
    print(normalized.shape)

    # Select first signal
    sample = normalized[0]

    # Create two augmented views
    view_1, view_2 = create_two_views(
        sample
    )

    print("\nOriginal sample shape:")
    print(sample.shape)

    print("\nView 1 shape:")
    print(view_1.shape)

    print("\nView 2 shape:")
    print(view_2.shape)

    print("\nLabel of sample:")
    print(labels[0])

    print("\nVelocity of sample:")
    print(round(velocities[0], 2), "m/s")

    print("\nPreprocessing successful!")