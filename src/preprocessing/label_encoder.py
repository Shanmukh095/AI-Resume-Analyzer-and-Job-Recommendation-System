import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def encode_labels(df, save_path):

    encoder = LabelEncoder()

    df["label"] = encoder.fit_transform(
        df["Category"]
    )

    save_path = Path(save_path)

    # Create parent directory if missing
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(save_path, "wb") as f:
        pickle.dump(encoder, f)

    return df
