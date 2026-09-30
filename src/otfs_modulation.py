import numpy as np

from otfs import create_otfs_grid
from config import N_DELAY, N_DOPPLER


def otfs_modulate(delay_doppler_grid):
    """
    Convert an OTFS delay-Doppler grid into
    a time-frequency grid and then into
    a time-domain signal.

    Parameters:
        delay_doppler_grid : 2D complex NumPy array

    Returns:
        time_frequency_grid : 2D complex array
        time_domain_signal  : 1D complex array
    """

    # ------------------------------------------------
    # Step 1: Delay-Doppler -> Time-Frequency
    # ------------------------------------------------

    time_frequency_grid = np.fft.ifft(
        np.fft.fft(delay_doppler_grid, axis=0),
        axis=1
    )

    # ------------------------------------------------
    # Step 2: Time-Frequency -> Time Domain
    # ------------------------------------------------

    time_domain_matrix = np.fft.ifft(
        time_frequency_grid,
        axis=0
    )

    # Convert 2D matrix into 1D transmitted signal
    time_domain_signal = time_domain_matrix.flatten()

    return time_frequency_grid, time_domain_signal


if __name__ == "__main__":

    print("===================================")
    print("       OTFS MODULATION TEST")
    print("===================================")

    # Create delay-Doppler grid
    delay_doppler_grid = create_otfs_grid()

    print("\nDelay-Doppler grid shape:")
    print(delay_doppler_grid.shape)

    # Perform OTFS modulation
    time_frequency_grid, time_domain_signal = otfs_modulate(
        delay_doppler_grid
    )

    print("\nTime-Frequency grid shape:")
    print(time_frequency_grid.shape)

    print("\nTime-domain signal shape:")
    print(time_domain_signal.shape)

    print("\nFirst 10 time-domain samples:")
    print(time_domain_signal[:10])

    print("\nNumber of transmitted samples:")
    print(len(time_domain_signal))

    print("\nOTFS modulation successful!")