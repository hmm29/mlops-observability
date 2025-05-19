# src/data_validation/schema.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import json

class DataSchemaValidator:
    def __init__(self, schema_path=None, schema=None):
        if schema:
            self.schema = schema
        elif schema_path:
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
        else:
            raise ValueError("Either schema or schema_path must be provided")
            
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data against schema and return validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "missing_columns": [],
            "type_errors": [],
            "range_errors": []
        }
        
        # Check required columns
        required_columns = [col for col, props in self.schema["features"].items() 
                           if props.get("required", False)]
        
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            results["valid"] = False
            results["missing_columns"] = missing
            results["errors"].append(f"Missing required columns: {', '.join(missing)}")
        
        # Check data types and ranges
        for col, props in self.schema["features"].items():
            if col not in data.columns:
                continue
                
            # Type validation
            dtype = props.get("type")
            if dtype == "numeric" and not pd.api.types.is_numeric_dtype(data[col]):
                results["valid"] = False
                results["type_errors"].append(col)
                results["errors"].append(f"Column {col} should be numeric")
            
            # Range validation
            if "range" in props and pd.api.types.is_numeric_dtype(data[col]):
                min_val, max_val = props["range"]
                if data[col].min() < min_val or data[col].max() > max_val:
                    results["valid"] = False
                    results["range_errors"].append(col)
                    results["errors"].append(
                        f"Column {col} has values outside range [{min_val}, {max_val}]"
                    )
        
        return results
