
import pandas as pd
import torch
import os

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings():

    print("Loading job dataset...")

    df = pd.read_csv(
        "data/processed(job matcher)/processed_jobs.csv"
    )

    print("Dataset shape:", df.shape)


    print("Loading Sentence-BERT model...")

    model = SentenceTransformer(MODEL_NAME)


    texts = df["clean_job_text"].tolist()


    print("Generating embeddings...")


    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True
    )


    os.makedirs(
        "saved_models/sentence_bert",
        exist_ok=True
    )


    torch.save(
        embeddings,
        "saved_models/sentence_bert/job_embeddings.pt"
    )


    df.to_csv(
        "saved_models/sentence_bert/job_metadata.csv",
        index=False
    )


    print("Job embeddings saved successfully")


if __name__ == "__main__":

    generate_embeddings()
