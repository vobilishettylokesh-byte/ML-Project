import numpy as np

from modulation import (
    generate_random_bits,
    qpsk_modulate
)

from config import N_DELAY, N_DOPPLER


def create_otfs_grid():
    """
    Generate QPSK symbols and arrange them
    into an OTFS delay-Doppler grid.
    """

    # Total number of QPSK symbols
    num_symbols = N_DELAY * N_DOPPLER

    # QPSK uses 2 bits per symbol
    num_bits = 2 * num_symbols

    # Generate random bits
    bits = generate_random_bits(num_bits)

    # Convert bits to QPSK symbols
    symbols = qpsk_modulate(bits)

    # Arrange symbols into a 2D delay-Doppler grid
    grid = symbols.reshape(N_DELAY, N_DOPPLER)

    return grid


if __name__ == "__main__":

    # Create OTFS grid
    grid = create_otfs_grid()

    print("===================================")
    print("          OTFS GRID TEST")
    print("===================================")

    print("\nGrid shape:")
    print(grid.shape)

    print("\nNumber of symbols:")
    print(grid.size)

    print("\nOTFS Delay-Doppler Grid:")
    print(grid)

    print("\nOTFS grid creation successful!")