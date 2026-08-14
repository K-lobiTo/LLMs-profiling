import pandas as pd
from pathlib import Path

QA_DIR = Path("qa_dataset")

QA_FILES = {
    "binary": QA_DIR / "yes_no.csv",
    "short_answer": QA_DIR / "short_answer.csv",
    "open_ended": QA_DIR / "open_ended.csv",
}


def load_qa_dataset(qa_dir=QA_DIR):
    """
    Loads all QA category CSVs into a dict of DataFrames.
    Returns: {"binary": df, "short_answer": df, "open_ended": df}
    """
    dataset = {}
    for category, filename in QA_FILES.items():
        path = Path(qa_dir) / filename.name
        df = pd.read_csv(path)
        dataset[category] = df
    return dataset


def load_qa_combined(qa_dir=QA_DIR):
    """
    Loads and concatenates all categories into a single DataFrame,
    useful for aggregate stats or shuffled sampling across categories.
    """
    dataset = load_qa_dataset(qa_dir)
    return pd.concat(dataset.values(), ignore_index=True)


if __name__ == "__main__":
    qa = load_qa_dataset()
    for category, df in qa.items():
        print(f"{category}: {len(df)} questions")

    # Example: iterate over a single category
    for _, row in qa["binary"].head(2).iterrows():
        print(row["Pregunta"], "->", row["Respuesta"])