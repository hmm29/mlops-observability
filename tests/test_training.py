import os
import tempfile

from src.data_validation.schema import DataSchemaValidator
from src.model.training import FEATURES, SCHEMA, generate_data, load_bundle, train_bundle
from tests.helpers import raises, trained_bundle, trained_bundle_dir


def test_generated_data_is_reproducible_and_valid():
    first, second = generate_data(rows=200, seed=5), generate_data(rows=200, seed=5)
    assert first.equals(second)
    assert list(first.columns) == FEATURES + ["flagged"]
    assert set(first["flagged"].unique()) <= {0, 1}
    assert DataSchemaValidator(schema=SCHEMA).validate(first[FEATURES])["valid"] is True


def test_drifted_data_has_larger_amounts():
    normal, drifted = generate_data(rows=500, seed=5), generate_data(rows=500, seed=5, drifted=True)
    assert drifted["amount"].median() > 2 * normal["amount"].median()


def test_bundle_has_every_file_and_loads():
    directory = trained_bundle_dir()
    assert sorted(os.listdir(directory)) == ["metadata.json", "model.joblib", "reference_data.csv", "schema.json"]
    bundle = trained_bundle()
    assert bundle.schema == SCHEMA
    assert list(bundle.reference_data.columns) == FEATURES
    assert bundle.metadata["data"] == "synthetic"


def test_model_beats_chance_on_held_out_data():
    # The label follows a known formula, so a working pipeline separates the classes.
    assert trained_bundle().metadata["metrics"]["roc_auc"] > 0.7


def test_model_scores_riskier_transactions_higher():
    bundle = trained_bundle()
    fresh = generate_data(rows=2, seed=1)[FEATURES]
    fresh.loc[0] = [15.0, 3000, 0, "grocery"]
    fresh.loc[1] = [4000.0, 5, 12, "electronics"]
    low, high = bundle.model.predict_proba(fresh)[:, 1]
    assert high > low


def test_loading_an_incomplete_bundle_fails_clearly():
    with tempfile.TemporaryDirectory() as directory:
        assert raises(FileNotFoundError, load_bundle, directory)


def test_training_is_deterministic():
    with tempfile.TemporaryDirectory() as directory:
        first = train_bundle(directory, rows=400, seed=9)["metrics"]
        second = train_bundle(directory, rows=400, seed=9)["metrics"]
    assert first == second
