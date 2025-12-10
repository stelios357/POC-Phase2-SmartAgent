#!/usr/bin/env python3
"""
Test script for the new batch analysis API with per-stock suffixes.
"""

import requests
import json

def test_batch_api():
    """Test the new batch API with different suffixes per stock."""

    # Test data with different suffixes
    test_stocks = [
        {"ticker": "AAPL", "suffix": "global"},
        {"ticker": "RELIANCE", "suffix": "nse"},
        {"ticker": "TCS", "suffix": "bse"}
    ]

    url = "http://localhost:5001/api/stocks/batch"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json={"stocks": test_stocks}, headers=headers)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Batch API call successful!")
                print(f"Retrieved data for {data.get('count', 0)} stocks")
                print("Stock tickers in response:", list(data.get("data", {}).keys()))
            else:
                print("❌ API returned error:", data.get("message"))
        else:
            print(f"❌ HTTP Error {response.status_code}")
            print("Response:", response.text)

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is the Flask app running on port 5001?")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_batch_api()