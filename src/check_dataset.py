import numpy as np


# Load dataset

signals = np.load("data/signals.npy")

labels = np.load("data/labels.npy")

velocities = np.load(
    "data/velocities.npy"
)


print("===================================")
print("        DATASET CHECK")
print("===================================")

print("\nSignals:")
print("Shape:", signals.shape)
print("Data type:", signals.dtype)

print("\nLabels:")
print("Shape:", labels.shape)
print("Unique classes:", np.unique(labels))

print("\nVelocities:")
print("Shape:", velocities.shape)

print("\nFirst sample:")
print(signals[0][:10])

print("\nFirst label:")
print(labels[0])

print("\nFirst velocity:")
print(velocities[0])

print("\nDataset check successful!")