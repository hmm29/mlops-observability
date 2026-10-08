"""Send synthetic requests to the API so the dashboards have something to show.

Usage:
    python -m scripts.simulate_traffic --normal 200 --drifted 200

Normal requests follow the training distribution. Drifted requests have larger
amounts and a different category mix, which the drift check should flag.
"""
import argparse
import json
import urllib.error
import urllib.request

from src.model.training import FEATURES, generate_data


def post(url: str, payload: dict) -> int:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def send(url: str, rows: int, drifted: bool, seed: int) -> dict:
    counts: dict = {}
    if rows <= 0:
        return counts
    data = generate_data(rows=rows, seed=seed, drifted=drifted)[FEATURES]
    for index, record in enumerate(json.loads(data.to_json(orient="records"))):
        status = post(f"{url}/predict", {"features": record, "request_id": f"sim-{seed}-{index}"})
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--normal", type=int, default=200)
    parser.add_argument("--drifted", type=int, default=200)
    parser.add_argument("--invalid", type=int, default=5, help="requests that should fail validation")
    args = parser.parse_args()

    print("normal: ", send(args.url, args.normal, drifted=False, seed=1))
    invalid = {}
    for index in range(args.invalid):
        status = post(f"{args.url}/predict", {"features": {"amount": "not a number"}, "request_id": f"bad-{index}"})
        invalid[status] = invalid.get(status, 0) + 1
    print("invalid:", invalid)
    print("drifted:", send(args.url, args.drifted, drifted=True, seed=2))

    with urllib.request.urlopen(f"{args.url}/drift", timeout=10) as response:
        report = json.loads(response.read()).get("report") or {}
    print("flagged features:", report.get("flagged_features", []))


if __name__ == "__main__":
    main()
