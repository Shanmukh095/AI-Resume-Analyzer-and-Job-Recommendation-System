# src/models/roberta/predict.py

import pickle
import torch

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification
)

from src.config import LABEL_ENCODER_PATH, MODEL_DIR

class ResumePredictor:

    def __init__(self):

        self.tokenizer = (
            RobertaTokenizer.from_pretrained(
                str(MODEL_DIR / "roberta")
            )
        )

        self.model = (
            RobertaForSequenceClassification.from_pretrained(
                str(MODEL_DIR / "roberta")
            )
        )

        with open(
            LABEL_ENCODER_PATH,
            "rb"
        ) as f:

            self.label_encoder = pickle.load(f)

    def predict(self, resume_text):

        inputs = self.tokenizer(
            resume_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence = torch.max(
            probabilities
        ).item()

        predicted_id = torch.argmax(
            probabilities,
            dim=1
        ).item()

        predicted_category = (
            self.label_encoder
            .inverse_transform([predicted_id])[0]
        )

        return {
            "category": predicted_category,
            "confidence": round(confidence, 4)
        }