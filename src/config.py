# ============================================
# OTFS Doppler Classification
# Configuration File
# ============================================

# OTFS grid size
N_DELAY = 16
N_DOPPLER = 16

# Modulation
MODULATION = "QPSK"

# Carrier frequency
CARRIER_FREQUENCY = 4e9       # 4 GHz

# Speed of light
SPEED_OF_LIGHT = 3e8          # m/s

# Doppler velocity ranges
LOW_VELOCITY = (0, 20)
MEDIUM_VELOCITY = (20, 80)
HIGH_VELOCITY = (80, 200)

# Signal-to-noise ratio
SNR_DB = 15

# Number of samples per class
SAMPLES_PER_CLASS = 1000

# Random seed
RANDOM_SEED = 42