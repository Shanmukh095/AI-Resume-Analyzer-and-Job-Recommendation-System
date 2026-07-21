# src/models/distilbert/evaluate.py

import numpy as np
import pandas as pd

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.config import TEST_CSV, MODEL_DIR
from src.models.distilbert.dataset import ResumeDataset


def evaluate_distilbert():

    test_df = pd.read_csv(TEST_CSV)

    test_df = test_df.dropna(subset=["text"])
    test_df["text"] = test_df["text"].astype(str)

    tokenizer = DistilBertTokenizer.from_pretrained(
        str(MODEL_DIR / "distilbert")
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        str(MODEL_DIR / "distilbert")
    )

    encodings = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=512
    )

    dataset = ResumeDataset(
        encodings,
        test_df["label"].tolist()
    )

    trainer = Trainer(model=model)

    predictions = trainer.predict(dataset)

    preds = np.argmax(
        predictions.predictions,
        axis=1
    )

    labels = test_df["label"].values

    print("=" * 60)
    print("DISTILBERT MODEL EVALUATION")
    print("=" * 60)

    print(f"Test Samples     : {len(test_df)}")
    print(f"Number of Classes: {test_df['label'].nunique()}")

    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            average="weighted"
        )
    )

    print("\n===== DISTILBERT RESULTS =====")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print(
        classification_report(
            labels,
            preds,
            digits=4
        )
    )

    print("\nConfusion Matrix")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    evaluate_distilbert()