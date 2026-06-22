import pandas as pd

from src.config import (
    RESUME_CSV,
    TRAIN_CSV,
    TEST_CSV,
    LABEL_ENCODER_PATH,
    TEST_SIZE,
    RANDOM_STATE
)
from src.config import PROCESSED_DIR
from src.preprocessing.text_cleaner import clean_resume
from src.preprocessing.label_encoder import encode_labels
from src.preprocessing.data_splitter import split_data


def run_preprocessing():
    

    print("Loading Resume Dataset...")

    PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
    )

    df = pd.read_csv(RESUME_CSV)

    print(f"Original Shape: {df.shape}")

    # Clean text
    df["clean_resume"] = (
        df["Resume_str"]
        .astype(str)
        .apply(clean_resume)
    )

    # Remove duplicates
    df = df.drop_duplicates(
        subset=["clean_resume"]
    )

    print(
        f"After Duplicate Removal: {df.shape}"
    )

    # Label Encoding
    df = encode_labels(
        df,
        LABEL_ENCODER_PATH
    )

    # Split
    X_train, X_test, y_train, y_test = split_data(
        df,
        TEST_SIZE,
        RANDOM_STATE
    )

    train_df = pd.DataFrame({
        "text": X_train,
        "label": y_train
    }).reset_index(drop=True)

    test_df = pd.DataFrame({
        "text": X_test,
        "label": y_test
    }).reset_index(drop=True)

    TRAIN_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_CSV,
        index=False
    )

    test_df.to_csv(
        TEST_CSV,
        index=False
    )

    print("Preprocessing Completed")

    print(
        f"Train Shape: {train_df.shape}"
    )

    print(
        f"Test Shape: {test_df.shape}"
    )
