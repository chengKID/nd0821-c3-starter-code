"""Send a POST request to the deployed (live) API and print the result."""
import os
import sys
from urllib.parse import urlparse

import requests

# Replace with your Render URL, or set API_URL in the environment.
API_URL = os.environ.get("API_URL", "https://census-income-api.onrender.com")

SAMPLE = {
    "age": 52,
    "workclass": "Self-emp-inc",
    "fnlgt": 287927,
    "education": "Doctorate",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 99999,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}


def main():
    host = urlparse(API_URL).hostname or ""
    if host in {"localhost", "127.0.0.1", "0.0.0.0"}:
        sys.exit(f"API_URL points to a local server ({API_URL}); use the live URL.")

    # Free Render instances sleep when idle; the first request can take ~1 min.
    response = requests.post(f"{API_URL.rstrip('/')}/predict", json=SAMPLE, timeout=120)

    print(f"Request URL: {response.url}")
    print(f"Status code: {response.status_code}")
    print(f"Model inference result: {response.json()}")


if __name__ == "__main__":
    main()
