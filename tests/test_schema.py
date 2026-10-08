import json
import os
import tempfile

import pandas as pd

from src.data_validation.schema import DataSchemaValidator
from src.model.training import SCHEMA
from tests.helpers import VALID_FEATURES, raises


def validate(**overrides):
    row = {**VALID_FEATURES, **overrides}
    row = {key: value for key, value in row.items() if value is not ...}
    return DataSchemaValidator(schema=SCHEMA).validate(pd.DataFrame([row]))


def test_valid_row_passes():
    result = validate()
    assert result["valid"] is True
    assert result["errors"] == []


def test_missing_required_column():
    result = validate(amount=...)
    assert result["valid"] is False
    assert result["missing_columns"] == ["amount"]


def test_null_in_required_column():
    result = validate(amount=None)
    assert result["valid"] is False
    assert "amount" in result["missing_columns"]


def test_wrong_type_for_numeric_column():
    result = validate(amount="lots")
    assert result["type_errors"] == ["amount"]
    assert validate(amount=True)["type_errors"] == ["amount"]


def test_value_out_of_range():
    result = validate(amount=-5)
    assert result["range_errors"] == ["amount"]
    assert validate(amount=10**9)["valid"] is False


def test_unknown_category_and_wrong_category_type():
    assert validate(merchant_category="yachts")["range_errors"] == ["merchant_category"]
    assert validate(merchant_category=7)["type_errors"] == ["merchant_category"]


def test_every_problem_is_reported_together():
    result = validate(amount="x", account_age_days=..., merchant_category="yachts")
    assert len(result["errors"]) == 3


def test_schema_can_be_loaded_from_a_file():
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "schema.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(SCHEMA, file)
        validator = DataSchemaValidator(schema_path=path)
    assert validator.validate(pd.DataFrame([VALID_FEATURES]))["valid"] is True


def test_a_schema_is_required():
    assert raises(ValueError, DataSchemaValidator)
