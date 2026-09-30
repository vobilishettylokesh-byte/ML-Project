import numpy as np
import matplotlib.pyplot as plt

from otfs import create_otfs_grid
from otfs_modulation import otfs_modulate
from doppler import apply_doppler, calculate_doppler_frequency


# ============================================
# Generate OTFS signal
# ============================================

delay_doppler_grid = create_otfs_grid()

_, transmitted_signal = otfs_modulate(
    delay_doppler_grid
)


# ============================================
# Simulation parameters
# ============================================

velocity = 50          # m/s
sampling_rate = 10000  # Hz


# ============================================
# Calculate Doppler frequency
# ============================================

doppler_frequency = calculate_doppler_frequency(
    velocity
)

print("===================================")
print("       DOPPLER SIGNAL TEST")
print("===================================")

print("\nVelocity:")
print(velocity, "m/s")

print("\nDoppler frequency:")
print(round(doppler_frequency, 2), "Hz")


# ============================================
# Apply Doppler
# ============================================

received_signal = apply_doppler(
    transmitted_signal,
    velocity,
    sampling_rate
)


# ============================================
# Display results
# ============================================

print("\nTransmitted signal length:")
print(len(transmitted_signal))

print("\nReceived signal length:")
print(len(received_signal))

print("\nDoppler effect successfully applied!")


# ============================================
# Plot phase
# ============================================

plt.figure(figsize=(10, 5))

plt.plot(
    np.angle(transmitted_signal),
    label="Transmitted"
)

plt.plot(
    np.angle(received_signal),
    label="After Doppler"
)

plt.xlabel("Sample Index")

plt.ylabel("Phase (radians)")

plt.title("Doppler Effect on OTFS Signal")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()