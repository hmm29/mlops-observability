"""Shared test helpers."""
import tempfile
from functools import lru_cache

from src.model.training import ModelBundle, load_bundle, train_bundle

VALID_FEATURES = {
    "amount": 42.5,
    "account_age_days": 400,
    "transactions_last_24h": 2,
    "merchant_category": "grocery",
}


@lru_cache(maxsize=1)
def trained_bundle_dir() -> str:
    """Train the example model once per test run, into a temporary folder."""
    directory = tempfile.mkdtemp(prefix="bundle-")
    train_bundle(directory, rows=800, seed=42)
    return directory


def trained_bundle() -> ModelBundle:
    return load_bundle(trained_bundle_dir())


def raises(exception_type, function, *args, **kwargs) -> bool:
    try:
        function(*args, **kwargs)
    except exception_type:
        return True
    return False
