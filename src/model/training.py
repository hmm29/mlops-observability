"""A small example model so the monitoring has something real to watch.

The data is synthetic: transactions with four features and a "flagged" label
drawn from a fixed formula plus noise. Nothing here is trained on real data.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_NAME = "example_model"
NUMERIC_FEATURES = ["amount", "account_age_days", "transactions_last_24h"]
CATEGORICAL_FEATURES = ["merchant_category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CATEGORIES = ["grocery", "restaurants", "electronics", "travel", "other"]
CATEGORY_WEIGHTS = [0.35, 0.25, 0.15, 0.10, 0.15]
CATEGORY_RISK = {"grocery": -0.5, "restaurants": -0.3, "electronics": 0.8, "travel": 0.6, "other": 0.0}

SCHEMA: Dict[str, Any] = {
    "features": {
        "amount": {"type": "numeric", "required": True, "range": [0, 50000]},
        "account_age_days": {"type": "numeric", "required": True, "range": [0, 20000]},
        "transactions_last_24h": {"type": "numeric", "required": True, "range": [0, 1000]},
        "merchant_category": {"type": "categorical", "required": True, "allowed": CATEGORIES},
    }
}


def generate_data(rows: int = 2000, seed: int = 42, drifted: bool = False) -> pd.DataFrame:
    """Generate synthetic transactions.

    With `drifted=True` amounts are about three times larger and the category
    mix leans toward electronics and travel, to simulate production data that
    has moved away from the training data.
    """
    rng = np.random.default_rng(seed)
    weights = [0.10, 0.10, 0.40, 0.30, 0.10] if drifted else CATEGORY_WEIGHTS
    data = pd.DataFrame(
        {
            "amount": np.round(rng.lognormal(mean=5.1 if drifted else 4.0, sigma=0.9, size=rows).clip(0.5, 50000), 2),
            "account_age_days": rng.integers(1, 3650, size=rows),
            "transactions_last_24h": rng.poisson(3.0, size=rows),
            "merchant_category": rng.choice(CATEGORIES, size=rows, p=weights),
        }
    )
    logit = (
        -2.2
        + 1.1 * (np.log(data["amount"]) - 4.0)
        - 0.0006 * data["account_age_days"]
        + 0.25 * data["transactions_last_24h"]
        + data["merchant_category"].map(CATEGORY_RISK)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    data["flagged"] = (rng.random(rows) < probability).astype(int)
    return data


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("classifier", LogisticRegression(max_iter=1000))])


def train_bundle(output_dir: str, rows: int = 2000, seed: int = 42, version: str = "1") -> Dict[str, Any]:
    """Train the example model and write everything the API needs to `output_dir`.

    Files written: model.joblib, schema.json, reference_data.csv, metadata.json.
    Returns the metadata, including accuracy and ROC AUC on a held-out split.
    """
    data = generate_data(rows=rows, seed=seed)
    train, test = train_test_split(data, test_size=0.25, random_state=seed, stratify=data["flagged"])

    pipeline = build_pipeline()
    pipeline.fit(train[FEATURES], train["flagged"])
    probabilities = pipeline.predict_proba(test[FEATURES])[:, 1]
    metadata = {
        "model_name": MODEL_NAME,
        "version": version,
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "seed": seed,
        "data": "synthetic",
        "metrics": {
            "accuracy": round(float(accuracy_score(test["flagged"], probabilities >= 0.5)), 4),
            "roc_auc": round(float(roc_auc_score(test["flagged"], probabilities)), 4),
        },
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "model.joblib")
    (out / "schema.json").write_text(json.dumps(SCHEMA, indent=2), encoding="utf-8")
    train[FEATURES].to_csv(out / "reference_data.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


@dataclass
class ModelBundle:
    model: Any
    schema: Dict[str, Any]
    reference_data: pd.DataFrame
    metadata: Dict[str, Any]


def load_bundle(bundle_dir: str) -> ModelBundle:
    """Load a bundle written by `train_bundle`. Raises FileNotFoundError if incomplete."""
    path = Path(bundle_dir)
    required = ["model.joblib", "schema.json", "reference_data.csv", "metadata.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Model bundle at {path} is missing: {', '.join(missing)}")
    return ModelBundle(
        model=joblib.load(path / "model.joblib"),
        schema=json.loads((path / "schema.json").read_text(encoding="utf-8")),
        reference_data=pd.read_csv(path / "reference_data.csv"),
        metadata=json.loads((path / "metadata.json").read_text(encoding="utf-8")),
    )


def default_bundle_dir(model_name: Optional[str] = None) -> str:
    return str(Path("models") / (model_name or MODEL_NAME))
