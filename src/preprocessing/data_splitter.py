from sklearn.model_selection import train_test_split


def split_data(
    df,
    test_size,
    random_state
):

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_resume"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )

    return X_train, X_test, y_train, y_test
