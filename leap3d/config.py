import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Data configuration
DATA_DIR = Path(os.getenv("DATA_DIR"))
DATASET_DIR = Path(os.getenv("DATASET_DIR"))
PARAMS_FILEPATH = Path(os.getenv("PARAMS_FILEPATH"))
ROUGH_COORDS_FILEPATH = Path(os.getenv("ROUGH_COORDS_FILEPATH"))

SCALE_DISTANCE_BY = float(os.getenv("SCALE_DISTANCE_BY", 1))
SCALE_TIME_BY = float(os.getenv("SCALE_TIME_BY", 1))

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
BASE_TEMPERATURE = float(os.getenv("BASE_TEMPERATURE"))
MAX_LASER_POWER = float(os.getenv("MAX_LASER_POWER"))
MAX_LASER_RADIUS = float(os.getenv("MAX_LASER_RADIUS"))

# Device settings
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 1))
FORCE_PREPARE = os.getenv("FORCE_PREPARE", False) == "True"
