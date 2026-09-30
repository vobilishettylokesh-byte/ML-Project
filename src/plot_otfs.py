import numpy as np
import matplotlib.pyplot as plt

from otfs import create_otfs_grid
from otfs_modulation import otfs_modulate


# Create OTFS grid
delay_doppler_grid = create_otfs_grid()

# Modulate
time_frequency_grid, time_domain_signal = otfs_modulate(
    delay_doppler_grid
)

# Time samples
samples = np.arange(len(time_domain_signal))

# Magnitude of signal
signal_magnitude = np.abs(time_domain_signal)


# Plot
plt.figure(figsize=(10, 5))

plt.plot(samples, signal_magnitude)

plt.xlabel("Sample Index")
plt.ylabel("Signal Magnitude")

plt.title("OTFS Time-Domain Transmitted Signal")

plt.grid(True)

plt.tight_layout()

plt.show()