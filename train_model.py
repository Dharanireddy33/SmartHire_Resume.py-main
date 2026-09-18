import json
import os
import shutil
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import numpy as np
except Exception:
    np = None


def sanitize_for_json(obj):
    """Recursively convert numpy/pandas types to native Python types for JSON."""
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    try:
        # pandas scalar types
        import pandas as _pd

        if isinstance(obj, (_pd.Int64Dtype().type,)):
            return int(obj)
    except Exception:
        pass
    return obj


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def find_file_in_ancestors(start_dir: str, relative_path: str) -> str | None:
    candidate = os.path.join(start_dir, relative_path)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    current_dir = start_dir
    while True:
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir or not parent_dir:
            break
        candidate = os.path.join(parent_dir, relative_path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        current_dir = parent_dir

    return None


def resolve_dataset_path():
    env_path = os.getenv("DATA_PATH") or os.getenv("DATA_URL") or os.getenv("RESUME_DATASET_PATH")
    if not env_path:
        env_path = "ResumeDataset.csv.csv"

    if env_path.startswith(("http://", "https://")):
        try:
            return download_dataset_from_url(env_path)
        except Exception as e:
            # Download failed (404, network, etc). Log and allow caller to fall back to sample data.
            print(f"Warning: failed to download dataset from {env_path}: {e}")
            return ""

    if os.path.isabs(env_path):
        return env_path

    found = find_file_in_ancestors(BASE_DIR, env_path)
    if found:
        return found

    return os.path.join(BASE_DIR, env_path)


def download_dataset_from_url(url: str) -> str:
    local_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
    os.makedirs(local_dir, exist_ok=True)
    filename = os.path.basename(url.split("?")[0]) or "ResumeDataset.csv"
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with urlopen(url, timeout=30) as response, open(local_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except HTTPError as he:
        print(f"Warning: HTTP error while downloading dataset: {he.code} {he.reason}")
        return ""
    except URLError as ue:
        print(f"Warning: URL error while downloading dataset: {ue}")
        return ""
    except Exception as e:
        print(f"Warning: unexpected error while downloading dataset: {e}")
        return ""

    return local_path


def load_sample_dataset() -> pd.DataFrame:
    sample_records = [
        {
            "Resume": "Experienced data scientist skilled in Python, pandas, scikit-learn, machine learning, and data analysis.",
            "Category": "Data Scientist",
        },
        {
            "Resume": "Software engineer with expertise in Java, REST APIs, Docker, and cloud-native application development.",
            "Category": "Software Engineer",
        },
        {
            "Resume": "Web developer experienced in HTML, CSS, JavaScript, React, and responsive front-end design.",
            "Category": "Web Developer",
        },
        {
            "Resume": "Business analyst with strong Excel, SQL, stakeholder communication, and process improvement skills.",
            "Category": "Business Analyst",
        },
        {
            "Resume": "Machine learning engineer with experience building predictive models, feature engineering, and model deployment.",
            "Category": "Machine Learning Engineer",
        },
    ]
    return pd.DataFrame(sample_records)


def train_and_evaluate_model():
    """Train a TF-IDF + Logistic Regression model for resume category classification."""
    dataset_path = resolve_dataset_path()
    if not os.path.exists(dataset_path):
        print(
            f"Resume dataset not found at '{dataset_path}'. Falling back to a sample dataset. "
            "Set DATA_PATH, DATA_URL, or RESUME_DATASET_PATH to use your own data."
        )
        df = load_sample_dataset()
    else:
        df = pd.read_csv(dataset_path)

    if "Resume" not in df.columns or "Category" not in df.columns:
        raise ValueError("The dataset must contain 'Resume' and 'Category' columns.")

    resumes = df["Resume"].fillna("").astype(str)
    categories = df["Category"].fillna("").astype(str)

    # Convert raw resume text into TF-IDF features.
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(resumes)

    # Encode category labels for model training.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(categories)

    # Split data into training and testing sets.
    # Only stratify when every class has at least 2 samples; otherwise fall back
    # to a plain split to avoid ValueError from sklearn.
    try:
        y_counts = pd.Series(y).value_counts()
        stratify_arg = y if (y_counts >= 2).all() else None
    except Exception:
        stratify_arg = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_arg,
    )

    # Train a classical logistic regression classifier.
    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set for evaluation.
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics.
    # Use only labels present in the test set when creating reports to avoid
    # mismatches between target_names and actual classes in y_test.
    labels_present = sorted(set(y_test))
    try:
        target_names_present = label_encoder.inverse_transform(labels_present)
    except Exception:
        target_names_present = [str(l) for l in labels_present]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted")), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted")), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels_present).tolist(),
        "classification_report": classification_report(y_test, y_pred, labels=labels_present, target_names=target_names_present),
        # `labels` contains the human-readable class names present in the report
        "labels": list(target_names_present),
        # `label_indices` contains the numeric label indices corresponding to `labels`
        "label_indices": [int(x) for x in labels_present],
    }

    # Save trained artifacts and evaluation results.
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))

    with open(os.path.join(model_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(sanitize_for_json(metrics), handle, indent=2)

    print("Model trained successfully with Logistic Regression.")
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1-Score:", metrics["f1_score"])
    print("Classification Report:\n", metrics["classification_report"])

    return metrics


if __name__ == "__main__":
    train_and_evaluate_model()