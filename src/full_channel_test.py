import numpy as np

from otfs import create_otfs_grid
from otfs_modulation import otfs_modulate
from channel import wireless_channel
from config import CARRIER_FREQUENCY, SNR_DB


# ============================================
# Parameters
# ============================================

VELOCITY = 50
SAMPLING_RATE = 10000


# ============================================
# Generate OTFS grid
# ============================================

delay_doppler_grid = create_otfs_grid()


# ============================================
# OTFS modulation
# ============================================

_, transmitted_signal = otfs_modulate(
    delay_doppler_grid
)


# ============================================
# Wireless channel
# ============================================

received_signal = wireless_channel(
    transmitted_signal,
    velocity=VELOCITY,
    sampling_rate=SAMPLING_RATE,
    carrier_frequency=CARRIER_FREQUENCY,
    snr_db=SNR_DB
)


# ============================================
# Display results
# ============================================

print("==========================================")
print("       COMPLETE OTFS CHANNEL TEST")
print("==========================================")

print("\nOTFS grid:")
print(delay_doppler_grid.shape)

print("\nTransmitted signal:")
print(transmitted_signal.shape)

print("\nVelocity:")
print(VELOCITY, "m/s")

print("\nSNR:")
print(SNR_DB, "dB")

print("\nReceived signal:")
print(received_signal.shape)

print("\nComplete OTFS channel simulation successful!")