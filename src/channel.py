import numpy as np

from doppler import calculate_doppler_frequency


def add_awgn(signal, snr_db):
    """
    Add Additive White Gaussian Noise (AWGN)
    to a complex signal.

    Parameters:
        signal : complex input signal
        snr_db : signal-to-noise ratio in dB

    Returns:
        noisy signal
    """

    # Calculate average signal power
    signal_power = np.mean(np.abs(signal) ** 2)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise power
    noise_power = signal_power / snr_linear

    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal))
        + 1j * np.random.randn(len(signal))
    )

    # Add noise
    noisy_signal = signal + noise

    return noisy_signal


def apply_multipath(signal):
    """
    Simulate a simple multipath wireless channel.

    Three paths are created with different:
        - delays
        - channel gains
    """

    # Path delays in samples
    delays = [0, 2, 5]

    # Complex path gains
    gains = [
        1.0 + 0j,
        0.5 + 0.2j,
        0.25 - 0.1j
    ]

    # Output signal
    output = np.zeros(
        len(signal) + max(delays),
        dtype=np.complex64
    )

    # Apply each path
    for delay, gain in zip(delays, gains):

        output[delay:delay + len(signal)] += (
            gain * signal
        )

    return output


def apply_doppler(signal, velocity, sampling_rate, carrier_frequency):
    """
    Apply Doppler frequency shift.
    """

    # Doppler frequency
    doppler_frequency = (
        velocity * carrier_frequency
    ) / 3e8

    # Sample indices
    n = np.arange(len(signal))

    # Time
    time = n / sampling_rate

    # Doppler phase rotation
    phase = np.exp(
        1j * 2 * np.pi * doppler_frequency * time
    )

    # Apply Doppler
    return signal * phase


def wireless_channel(
    signal,
    velocity,
    sampling_rate,
    carrier_frequency,
    snr_db
):
    """
    Complete wireless channel.

    Order:

        Transmitted signal
              ↓
          Multipath
              ↓
           Doppler
              ↓
            AWGN
              ↓
        Received signal
    """

    # ----------------------------------------
    # 1. Multipath
    # ----------------------------------------

    multipath_signal = apply_multipath(signal)

    # ----------------------------------------
    # 2. Doppler
    # ----------------------------------------

    doppler_signal = apply_doppler(
        multipath_signal,
        velocity,
        sampling_rate,
        carrier_frequency
    )

    # ----------------------------------------
    # 3. AWGN
    # ----------------------------------------

    received_signal = add_awgn(
        doppler_signal,
        snr_db
    )

    return received_signal


if __name__ == "__main__":

    print("===================================")
    print("       WIRELESS CHANNEL TEST")
    print("===================================")

    # Create a simple test signal
    np.random.seed(42)

    signal = (
        np.random.randn(256)
        + 1j * np.random.randn(256)
    )

    # Channel parameters
    velocity = 50
    sampling_rate = 10000
    carrier_frequency = 4e9
    snr_db = 15

    # Apply complete channel
    received_signal = wireless_channel(
        signal,
        velocity,
        sampling_rate,
        carrier_frequency,
        snr_db
    )

    # Calculate Doppler frequency
    doppler_frequency = calculate_doppler_frequency(
        velocity
    )

    print("\nVelocity:")
    print(velocity, "m/s")

    print("\nDoppler frequency:")
    print(round(doppler_frequency, 2), "Hz")

    print("\nOriginal signal length:")
    print(len(signal))

    print("\nReceived signal length:")
    print(len(received_signal))

    print("\nSNR:")
    print(snr_db, "dB")

    print("\nMultipath + Doppler + AWGN successful!")