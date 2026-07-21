
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_model():

    model = SentenceTransformer(MODEL_NAME)

    return model


if __name__ == "__main__":

    model = load_model()

    print("Sentence-BERT model loaded successfully")
