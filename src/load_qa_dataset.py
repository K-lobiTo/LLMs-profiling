import pandas as pd
from pathlib import Path

QA_DIR = Path("data/qa_dataset")

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


def sample_qa(dataset, n=20, seed=None):
    """
    Takes a random sample of n rows from each category.

    Args:
        dataset: dict of {category: DataFrame}, e.g. output of load_qa_dataset()
        n: number of rows to sample per category (default 20)
        seed: optional int for reproducible sampling

    Returns:
        dict of {category: sampled DataFrame}, each with n rows
        (or fewer, if a category has less than n rows available)
    """
    sampled = {}
    for category, df in dataset.items():
        k = min(n, len(df))
        sampled[category] = df.sample(n=k, random_state=seed).reset_index(drop=True)
    return sampled


if __name__ == "__main__":
    qa = load_qa_dataset()
    for category, df in qa.items():
        print(f"{category}: {len(df)} questions")

    # Example: sample 20 random rows from each category
    sample = sample_qa(qa, n=20, seed=42)
    for category, df in sample.items():
        print(f"{category} sample: {len(df)} rows")

    # Example: iterate over a single sampled category
    for _, row in sample["binary"].head(2).iterrows():
        print(row["Pregunta"], "->", row["Respuesta"])