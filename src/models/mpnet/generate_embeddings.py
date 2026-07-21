
import os
import torch
import pandas as pd

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-mpnet-base-v2"


def generate_embeddings():

    print("Loading job dataset...")

    df = pd.read_csv(
        "data/processed(job matcher)/processed_jobs.csv"
    )

    print(f"Dataset Shape: {df.shape}")


    print("Loading MPNet model...")

    model = SentenceTransformer(MODEL_NAME)


    print("Generating job embeddings...")


    embeddings = model.encode(
        df["clean_job_text"].tolist(),
        batch_size=16,
        show_progress_bar=True,
        convert_to_tensor=True
    )


    os.makedirs(
        "saved_models/mpnet",
        exist_ok=True
    )


    torch.save(
        embeddings,
        "saved_models/mpnet/job_embeddings.pt"
    )


    df.to_csv(
        "saved_models/mpnet/job_metadata.csv",
        index=False
    )


    print("\nEmbeddings generated successfully!")
    print("Saved:")
    print(" - job_embeddings.pt")
    print(" - job_metadata.csv")


if __name__ == "__main__":

    generate_embeddings()
