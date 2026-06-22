from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

ZIP_PATH = DATA_DIR / "resume.zip"

EXTRACTED_DIR = DATA_DIR / "extracted"

PROCESSED_DIR = DATA_DIR / "processed"

RESUME_CSV = EXTRACTED_DIR / "Resume" / "Resume.csv"

TRAIN_CSV = PROCESSED_DIR / "train_resume.csv"

TEST_CSV = PROCESSED_DIR / "test_resume.csv"

LABEL_ENCODER_PATH = PROCESSED_DIR / "label_encoder.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42
