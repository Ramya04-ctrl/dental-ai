"""
config.py  -  ONE place for all the settings of the project.

If you want to change something (image size, number of epochs, blur limit...),
change it here and every other file will use the new value.
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Folders
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# WORK_DIR is where data, models, reports and the database live.
# Normally it is the project folder itself. (run_toy_demo.py changes it so the
# quick test never touches your real data.)
WORK_DIR = Path(os.environ.get("DENTAL_WORKDIR", BASE_DIR))

RAW_DIR = WORK_DIR / "dataset_raw"      # you put your original images here (one folder per class)
DATASET_DIR = WORK_DIR / "dataset"      # created automatically: train / val / test
DATA_DIR = WORK_DIR / "data"            # csv files for the ML risk model
MODELS_DIR = WORK_DIR / "models"        # trained models are saved here
REPORTS_DIR = WORK_DIR / "reports"      # charts and evaluation results
DB_PATH = WORK_DIR / "database" / "dental_screening.db"

MODEL_INFO_PATH = MODELS_DIR / "model_info.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
RISK_MODEL_PATH = MODELS_DIR / "risk_model.pkl"
RISK_INFO_PATH = MODELS_DIR / "risk_info.json"

# ----------------------------------------------------------------------------
# Classes (folder names inside dataset_raw/). Names must match your folders.
# ----------------------------------------------------------------------------
DEFAULT_CLASSES = [
    "caries", "discoloration", "fracture",
    "gingivitis", "healthy", "plaque_calculus",
]

CLASS_DISPLAY = {
    "caries": "Dental Caries (Tooth Decay)",
    "discoloration": "Tooth Discoloration",
    "fracture": "Tooth Fracture",
    "gingivitis": "Gingivitis",
    "healthy": "Healthy / No Visible Abnormality",
    "plaque_calculus": "Plaque and Calculus",
}

# ----------------------------------------------------------------------------
# Image settings
# ----------------------------------------------------------------------------
IMG_SIZE = (224, 224)                    # every image is resized to 224 x 224
# accepted file types: extension -> the REAL image format that must be inside the file
ALLOWED_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
ALLOWED_EXTENSIONS = set(ALLOWED_FORMATS)
MAX_FILE_MB = 10
MIN_SIDE_PIXELS = 200                    # image must be at least 200 px wide and tall
BLUR_THRESHOLD = 40.0                    # lower than this = too blurry (tune with YOUR images)
MIN_BRIGHTNESS = 40                      # average brightness (0-255) lower than this = too dark
MAX_BRIGHTNESS = 225                     # higher than this = too bright

# ----------------------------------------------------------------------------
# Deep-learning training settings
# ----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 16
EPOCHS = 15                  # rounds of training for the first stage
FINE_TUNE_EPOCHS = 5         # extra rounds for transfer-learning models
FINE_TUNE_LAYERS = 30        # how many top layers of the pretrained model are re-trained
SPLIT = (0.70, 0.15, 0.15)   # train / validation / test
USE_PRETRAINED = os.environ.get("DENTAL_NO_PRETRAINED", "0") != "1"

# If the model's highest probability is below this, we say "not sure"
CONFIDENCE_THRESHOLD = 0.50

# ----------------------------------------------------------------------------
# ML risk model settings
# ----------------------------------------------------------------------------
SYMPTOM_FEATURES = ["pain_level", "sensitivity", "gum_bleeding", "swelling", "symptom_days"]
RISK_LABELS = ["low", "moderate", "high"]
RISK_TARGET_COLUMN = "risk_label"


# Public online demo: set the secret/env value DENTAL_PUBLIC_DEMO=1 to switch OFF the saved
# history (on a public website the history file would be shared by every visitor).
PUBLIC_DEMO = os.environ.get("DENTAL_PUBLIC_DEMO", "0") == "1"


def ensure_dirs():
    """Create the folders if they do not exist yet."""
    for folder in (RAW_DIR, DATASET_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR, DB_PATH.parent):
        folder.mkdir(parents=True, exist_ok=True)


def display_name(class_name: str) -> str:
    """Nice name for a class folder name."""
    return CLASS_DISPLAY.get(class_name, class_name.replace("_", " ").title())
