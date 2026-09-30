import numpy as np

from config import CARRIER_FREQUENCY, SPEED_OF_LIGHT


def calculate_doppler_frequency(velocity):
    """
    Calculate Doppler frequency.

    Formula:
        fD = (v * fc) / c

    Parameters:
        velocity : velocity in m/s

    Returns:
        Doppler frequency in Hz
    """

    doppler_frequency = (
        velocity * CARRIER_FREQUENCY
    ) / SPEED_OF_LIGHT

    return doppler_frequency


def apply_doppler(signal, velocity, sampling_rate):
    """
    Apply Doppler frequency shift to a signal.

    Parameters:
        signal        : complex transmitted signal
        velocity      : relative velocity in m/s
        sampling_rate: sampling frequency in Hz

    Returns:
        Doppler-affected complex signal
    """

    # Calculate Doppler frequency
    doppler_frequency = calculate_doppler_frequency(velocity)

    # Create sample indices
    n = np.arange(len(signal))

    # Time for each sample
    time = n / sampling_rate

    # Doppler phase rotation
    doppler_phase = np.exp(
        1j * 2 * np.pi * doppler_frequency * time
    )

    # Apply Doppler effect
    received_signal = signal * doppler_phase

    return received_signal


def get_doppler_regime(velocity):
    """
    Classify velocity into a Doppler regime.

    Returns:
        0 -> Low
        1 -> Medium
        2 -> High
    """

    if velocity < 20:
        return 0, "Low Doppler"

    elif velocity < 80:
        return 1, "Medium Doppler"

    else:
        return 2, "High Doppler"


if __name__ == "__main__":

    print("===================================")
    print("       DOPPLER CHANNEL TEST")
    print("===================================")

    # Test velocities in m/s
    velocities = [
        10,
        50,
        100
    ]

    for velocity in velocities:

        # Calculate Doppler frequency
        doppler_frequency = calculate_doppler_frequency(
            velocity
        )

        # Get regime
        class_id, regime = get_doppler_regime(
            velocity
        )

        print("\nVelocity:", velocity, "m/s")

        print(
            "Doppler frequency:",
            round(doppler_frequency, 2),
            "Hz"
        )

        print("Regime:", regime)

        print("Class ID:", class_id)

    print("\nDoppler channel test successful!")