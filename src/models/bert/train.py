# src/models/bert/train.py

import pandas as pd

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from src.config import (
    TRAIN_CSV,
    TEST_CSV,
    MODEL_DIR
)

from src.models.bert.dataset import ResumeDataset


def train_bert():

    print("Loading processed datasets...")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Remove rows with missing resume text
    train_df = train_df.dropna(subset=["text"])
    test_df = test_df.dropna(subset=["text"])

    # Ensure text column contains strings
    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)

    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape : {test_df.shape}")

    print("Loading tokenizer...")

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=512
    )

    test_encodings = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=512
    )

    train_dataset = ResumeDataset(
        train_encodings,
        train_df["label"].tolist()
    )

    test_dataset = ResumeDataset(
        test_encodings,
        test_df["label"].tolist()
    )

    num_labels = train_df["label"].nunique()

    print("Loading BERT model...")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "bert"),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    print("Training started...")

    trainer.train()

    trainer.save_model(
        str(MODEL_DIR / "bert")
    )

    tokenizer.save_pretrained(
        str(MODEL_DIR / "bert")
    )

    print("Model saved successfully")


if __name__ == "__main__":
    train_bert()