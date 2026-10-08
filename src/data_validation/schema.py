"""Validate incoming feature rows against a JSON schema."""
import json
from typing import Any, Dict, Optional

import pandas as pd


class DataSchemaValidator:
    """Check required columns, types, numeric ranges and allowed categories.

    Schema format:

        {"features": {
            "amount": {"type": "numeric", "required": true, "range": [0, 50000]},
            "merchant_category": {"type": "categorical", "required": true,
                                  "allowed": ["grocery", "travel"]}
        }}
    """

    def __init__(self, schema_path: Optional[str] = None, schema: Optional[Dict[str, Any]] = None):
        if schema:
            self.schema = schema
        elif schema_path:
            with open(schema_path, "r", encoding="utf-8") as file:
                self.schema = json.load(file)
        else:
            raise ValueError("Either schema or schema_path must be provided")

    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "missing_columns": [],
            "type_errors": [],
            "range_errors": [],
        }

        def fail(kind: str, column: str, message: str) -> None:
            results["valid"] = False
            results[kind].append(column)
            results["errors"].append(message)

        for column, props in self.schema["features"].items():
            required = props.get("required", False)
            if column not in data.columns:
                if required:
                    fail("missing_columns", column, f"Missing required column: {column}")
                continue

            values = data[column]
            if required and values.isna().any():
                fail("missing_columns", column, f"Column {column} has missing values")
                continue
            values = values.dropna()
            if values.empty:
                continue

            kind = props.get("type")
            if kind == "numeric":
                is_number = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)
                if not is_number:
                    fail("type_errors", column, f"Column {column} should be numeric")
                    continue
                if "range" in props:
                    low, high = props["range"]
                    if values.min() < low or values.max() > high:
                        fail("range_errors", column, f"Column {column} has values outside range [{low}, {high}]")
            elif kind == "categorical":
                if not values.map(lambda value: isinstance(value, str)).all():
                    fail("type_errors", column, f"Column {column} should be text")
                    continue
                allowed = props.get("allowed")
                if allowed is not None:
                    unknown = sorted(set(values) - set(allowed))
                    if unknown:
                        fail("range_errors", column, f"Column {column} has unknown values: {', '.join(unknown)}")

        return results
