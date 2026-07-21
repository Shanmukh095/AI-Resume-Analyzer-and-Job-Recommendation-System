
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-mpnet-base-v2"


def load_model():

    print("Loading MPNet model...")

    model = SentenceTransformer(MODEL_NAME)

    print("MPNet model loaded successfully")

    return model


if __name__ == "__main__":

    load_model()
