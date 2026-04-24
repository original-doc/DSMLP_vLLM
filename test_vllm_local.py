#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
PROMPT = os.environ.get("PROMPT", "Explain in 3 sentences what a GPU MIG slice is.")


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": "Bearer dummy"}, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    try:
        models = get_json(f"{BASE_URL}/v1/models")
        model_id = models["data"][0]["id"]
        print(f"Using model: {model_id}")

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are a concise helpful assistant."},
                {"role": "user", "content": PROMPT},
            ],
            "temperature": 0.2,
            "max_tokens": 256,
        }
        out = post_json(f"{BASE_URL}/v1/chat/completions", payload)
        print("\nResponse:\n")
        print(out["choices"][0]["message"]["content"])
        return 0
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTPError {e.code}: {body}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
