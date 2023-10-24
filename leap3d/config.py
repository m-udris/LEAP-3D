import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Data configuration
DATA_DIR = Path(os.getenv("DATA_DIR"))
DATASET_DIR = Path(os.getenv("DATASET_DIR"))

# Scanning constants
TIMESTEP_DURATION = float(os.getenv("TIMESTEP_DURATION"))

X_MIN = float(os.getenv("X_MIN"))
X_MAX = float(os.getenv("X_MAX"))
Y_MIN = float(os.getenv("Y_MIN"))
Y_MAX = float(os.getenv("Y_MAX"))
Z_MIN = float(os.getenv("Z_MIN"))
Z_MAX = float(os.getenv("Z_MAX"))
SAFETY_OFFSET = float(os.getenv("SAFETY_OFFSET"))

MELTING_POINT = float(os.getenv("MELTING_POINT"))
