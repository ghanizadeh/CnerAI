"""
App-wide constants and configuration.
Edit here to change behaviour across the entire app.
"""

# ── Data ──────────────────────────────────────────────────────────────
MAX_UPLOAD_MB: int = 200
MAX_ROWS_PREVIEW: int = 1_000          # rows shown in st.dataframe
SUPPORTED_FILE_TYPES: list[str] = ["csv", "xls", "xlsx"]

# ── Modelling ─────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_CV_FOLDS: int = 5

# ── Outlier detection ─────────────────────────────────────────────────
DEFAULT_IQR_FACTOR: float = 1.5
DEFAULT_ZSCORE_THRESHOLD: float = 3.0
MAX_HEATMAP_FEATURES: int = 15        # above this, skip correlation heatmap

# ── UI ────────────────────────────────────────────────────────────────
APP_TITLE: str = "CnerAI"
APP_ICON: str = "🤖"
LOGO_PATH: str = "assets/logo.png"
