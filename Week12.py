"""Decision tree classification for the Iris dataset.

This script loads the local Iris CSV, trains a decision tree classifier, and
prints accuracy, precision, and recall for the test split.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_iris_data() -> pd.DataFrame:
    """Load the Iris dataset from the repository dataset folder."""

    csv_path = Path(__file__).with_name("dataset").joinpath("Iris.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {csv_path}")

    return pd.read_csv(csv_path)


def main() -> None:
    df = load_iris_data()

    feature_columns = [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ]

    X = df[feature_columns]
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print("Decision Tree Classifier on Iris Dataset")
    print("---------------------------------------")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nPredictions for test set:\n")

    results = pd.DataFrame(
        {
            "Actual": y_test.reset_index(drop=True),
            "Predicted": pd.Series(y_pred),
        }
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
