import numpy as np


def generate_random_bits(num_bits):
    """
    Generate random binary bits.

    Parameters:
        num_bits: Number of bits to generate

    Returns:
        NumPy array containing 0s and 1s
    """

    return np.random.randint(0, 2, num_bits)


def qpsk_modulate(bits):
    """
    Convert binary bits into QPSK complex symbols.

    Mapping:

        00 ->  1 + 1j
        01 ->  1 - 1j
        10 -> -1 + 1j
        11 -> -1 - 1j
    """

    bits = np.asarray(bits)

    # QPSK uses two bits per symbol
    if len(bits) % 2 != 0:
        raise ValueError("Number of bits must be even.")

    symbols = []

    for i in range(0, len(bits), 2):

        b1 = bits[i]
        b2 = bits[i + 1]

        if b1 == 0 and b2 == 0:
            symbol = 1 + 1j

        elif b1 == 0 and b2 == 1:
            symbol = 1 - 1j

        elif b1 == 1 and b2 == 0:
            symbol = -1 + 1j

        else:
            symbol = -1 - 1j

        symbols.append(symbol)

    return np.array(symbols, dtype=np.complex64)


if __name__ == "__main__":

    # Make the result reproducible
    np.random.seed(42)

    # Generate 20 bits
    bits = generate_random_bits(20)

    # QPSK modulation
    symbols = qpsk_modulate(bits)

    print("===================================")
    print("       QPSK MODULATION TEST")
    print("===================================")

    print("\nRandom Bits:")
    print(bits)

    print("\nQPSK Symbols:")
    print(symbols)

    print("\nNumber of bits:", len(bits))
    print("Number of symbols:", len(symbols))

    print("\nQPSK modulation successful!")