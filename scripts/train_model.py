"""Train the example model and write its bundle.

Usage:
    python -m scripts.train_model
    python -m scripts.train_model --register --tracking-uri sqlite:///mlflow.db

With --register the bundle is also uploaded to an MLflow registry and the new
version is moved to Production. That needs `pip install -r requirements-registry.txt`.
"""
import argparse
import json
import os

from src.model.training import MODEL_NAME, default_bundle_dir, train_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=default_bundle_dir())
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--register", action="store_true", help="also register the bundle in MLflow")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    args = parser.parse_args()

    metadata = train_bundle(args.out, rows=args.rows, seed=args.seed)
    print(f"Wrote model bundle to {args.out}")
    print(json.dumps(metadata["metrics"]))

    if args.register:
        from src.model_registry.client import ModelRegistry

        version = ModelRegistry(args.tracking_uri).log_bundle(
            args.out,
            MODEL_NAME,
            metrics=metadata["metrics"],
            tags={"data": "synthetic", "seed": str(args.seed)},
        )
        print(f"Registered {MODEL_NAME} version {version} in {args.tracking_uri} (stage: Production)")


if __name__ == "__main__":
    main()
