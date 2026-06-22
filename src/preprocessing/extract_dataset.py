import zipfile
import os

def extract_dataset(zip_path, extract_to):

    if os.path.exists(extract_to):
        print("Dataset already extracted")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    print("Dataset extracted successfully")
