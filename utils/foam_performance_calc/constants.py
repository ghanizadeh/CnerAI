"""
constants.py
------------
Central registry for all magic values, default settings, and string literals
used throughout the foam performance calculation pipeline.

Importing from here instead of hard-coding values makes future changes
(e.g. adding a new model, changing thresholds) a single-line edit.
"""

# ---------------------------------------------------------------------------
# Imputation model identifiers
# ---------------------------------------------------------------------------
MODEL_LINEAR: str = "linear"
MODEL_EXP: str = "exp"
MODEL_POLY: str = "poly"
MODEL_RF: str = "rf"
MODEL_BEST: str = "best"
MODEL_NONE: str = "none"

ALL_MODELS: list[str] = [MODEL_LINEAR, MODEL_EXP, MODEL_POLY, MODEL_RF]
MODEL_CHOICES: list[str] = [MODEL_BEST] + ALL_MODELS

# ---------------------------------------------------------------------------
# R² columns written by the imputer
# ---------------------------------------------------------------------------
R2_COLUMNS: list[str] = ["R2_linear", "R2_exp", "R2_poly", "R2_rf"]
BEST_MODEL_COLUMN: str = "Best_Model"

# ---------------------------------------------------------------------------
# Score output column names
# ---------------------------------------------------------------------------
COL_SCORE_VOLUME: str = "Score_Volume"
COL_SCORE_TEXTURE: str = "Score_Texture"
COL_PERFORMANCE_INDEX: str = "Performance_Index"
SCORE_COLUMNS: list[str] = [COL_SCORE_VOLUME, COL_SCORE_TEXTURE, COL_PERFORMANCE_INDEX]

# ---------------------------------------------------------------------------
# Texture weights CSV expected column names
# ---------------------------------------------------------------------------
TW_NORMALIZED_TEXTURE: str = "Normalized_Texture"
TW_WEIGHT: str = "Weight"

# ---------------------------------------------------------------------------
# Day-based renamed column templates
# ---------------------------------------------------------------------------
DAY_FOAM_TEMPLATE: str = "Day {day} - Foam (cc)"
DAY_TEXTURE_TEMPLATE: str = "Day {day} - Foam Texture"

# ---------------------------------------------------------------------------
# Column detection keywords
# ---------------------------------------------------------------------------
FOAM_CC_KEYWORD: str = "foam(cc)"
TEXTURE_KEYWORD: str = "texture"
SCORE_KEYWORD: str = "score"

# ---------------------------------------------------------------------------
# Imputation defaults
# ---------------------------------------------------------------------------
DEFAULT_NUM_DAYS: int = 14
DEFAULT_MODEL: str = MODEL_BEST
DEFAULT_R2_THRESHOLD: float = 0.0
DEFAULT_DAY_WEIGHT: float = 1.0

# ---------------------------------------------------------------------------
# Random Forest hyperparameters
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS: int = 200
RF_RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Polynomial degree
# ---------------------------------------------------------------------------
POLY_DEGREE: int = 2

# ---------------------------------------------------------------------------
# Download filename
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_FILENAME: str = "Foam_Performance_Index.csv"

# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
DAY_WEIGHT_COLUMNS: int = 7   # number of columns in the day-weight grid
MAX_NUM_DAYS: int = 14
MIN_NUM_DAYS: int = 1
