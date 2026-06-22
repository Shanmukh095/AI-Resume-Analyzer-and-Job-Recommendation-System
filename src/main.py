from src.config import ZIP_PATH, EXTRACTED_DIR

from src.preprocessing.extract_dataset import (
    extract_dataset
)

from src.preprocessing.preprocess_pipeline import (
    run_preprocessing
)


def main():

    print("Starting Project...")

    extract_dataset(
        zip_path=ZIP_PATH,
        extract_to=EXTRACTED_DIR
    )

    run_preprocessing()

    print("Pipeline Completed.")


if __name__ == "__main__":
    main()
