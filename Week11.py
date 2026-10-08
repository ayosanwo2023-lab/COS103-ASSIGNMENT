"""Exploratory data analysis for the Iris dataset.

This script loads the local Iris CSV, prints summary statistics, and saves a
few visualizations next to the script.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def load_iris_data() -> pd.DataFrame:
    """Load the Iris dataset from the repository dataset folder."""

    csv_path = Path(__file__).with_name("dataset").joinpath("Iris.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at {csv_path}")

    return pd.read_csv(csv_path)


def save_figure(filename: str) -> None:
    """Save the current Matplotlib figure next to this script."""

    output_path = Path(__file__).with_name(filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = load_iris_data()

    print("First 5 rows:\n")
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nColumn names:", list(df.columns))
    print("\nMissing values per column:\n")
    print(df.isna().sum())

    numeric_columns = [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
    ]

    print("\nSummary statistics for numeric columns:\n")
    print(df[numeric_columns].describe())
    print("\nSpecies counts:\n")
    print(df["Species"].value_counts())

    figure_width = 12
    figure_height = 8

    plt.figure(figsize=(figure_width, figure_height))
    for index, column in enumerate(numeric_columns, start=1):
        plt.subplot(2, 2, index)
        plt.hist(df[column], bins=15, color="#4C78A8", edgecolor="black")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
    save_figure("week11_histograms.png")

    plt.figure(figsize=(figure_width, 6))
    df.boxplot(column=numeric_columns, by="Species", grid=False)
    plt.suptitle("")
    plt.title("Feature Spread by Species")
    plt.xlabel("Species")
    plt.ylabel("Value")
    save_figure("week11_boxplots.png")

    plt.figure(figsize=(8, 6))
    colors = {
        "Iris-setosa": "#F58518",
        "Iris-versicolor": "#54A24B",
        "Iris-virginica": "#E45756",
    }
    for species, subset in df.groupby("Species"):
        plt.scatter(
            subset["SepalLengthCm"],
            subset["PetalLengthCm"],
            label=species,
            alpha=0.75,
            color=colors.get(species),
        )
    plt.title("Sepal Length vs Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend()
    save_figure("week11_scatter.png")

    plt.figure(figsize=(8, 6))
    correlation_matrix = df[numeric_columns].corr()
    plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=45, ha="right")
    plt.yticks(range(len(numeric_columns)), numeric_columns)
    plt.title("Correlation Heatmap")

    for row_index in range(len(numeric_columns)):
        for col_index in range(len(numeric_columns)):
            plt.text(
                col_index,
                row_index,
                f"{correlation_matrix.iloc[row_index, col_index]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    save_figure("week11_correlation_heatmap.png")

    print("\nPlots saved as:")
    print("- week11_histograms.png")
    print("- week11_boxplots.png")
    print("- week11_scatter.png")
    print("- week11_correlation_heatmap.png")


if __name__ == "__main__":
    main()
