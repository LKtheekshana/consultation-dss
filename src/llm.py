"""Small client for Llama 3 on a hosted, OpenAI-style API (Cloudflare Workers AI by default).

Used by Stage 5 (src/generate.py) and the web app. The API key is read from
the .env file (LLAMA_API_KEY=...) and is never written into the code.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


class LlamaError(RuntimeError):
    """Raised when the API cannot be used (no key, bad key, too many retries)."""


# Placeholder values from .env.example, which do not count as real settings
PLACEHOLDERS = {"", "your-key-here", "your-account-id-here"}


def env_value(name: str, env_file: Path = config.ENV_FILE) -> str | None:
    """Return a setting from the environment or the .env file, or None."""
    if os.environ.get(name):
        return os.environ[name].strip()
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name:
                value = value.strip().strip('"').strip("'")
                if value not in PLACEHOLDERS:
                    return value
    return None


def load_api_key(env_file: Path = config.ENV_FILE) -> str | None:
    """Return LLAMA_API_KEY from the environment or the .env file, or None."""
    return env_value("LLAMA_API_KEY", env_file)


def api_base_url(env_file: Path = config.ENV_FILE) -> str:
    """The API address. Cloudflare's address contains the account ID, which
    is read from CLOUDFLARE_ACCOUNT_ID in .env."""
    url = config.LLAMA_API_BASE_URL
    if "{account_id}" in url:
        account_id = env_value("CLOUDFLARE_ACCOUNT_ID", env_file)
        if not account_id:
            raise LlamaError("No Cloudflare account ID found. Add CLOUDFLARE_ACCOUNT_ID=... to .env "
                             "(it is shown on the Workers AI page of the Cloudflare dashboard).")
        url = url.replace("{account_id}", account_id)
    return url


def is_daily_limit(text: str) -> bool:
    """Cloudflare's free plan answers HTTP 429 with error 4006 when the daily
    allowance is used up. Waiting a few seconds will not help then."""
    text = text.lower()
    return "4006" in text or "daily free" in text or "neurons" in text


def chat(messages: list[dict], temperature: float, max_tokens: int,
         api_key: str | None = None, post=requests.post) -> dict:
    """Send a chat request and return {"text", "prompt_tokens", "completion_tokens", "seconds"}.

    messages: [{"role": "system" | "user", "content": "..."}]
    post:     the function that sends the HTTP request (the tests pass a fake one).

    If the rate limit is hit (HTTP 429) or the server has a temporary problem
    (HTTP 5xx), the request is retried after a pause that doubles each time.
    """
    api_key = api_key or load_api_key()
    if not api_key:
        raise LlamaError(
            f"No API key found. Copy .env.example to .env and set LLAMA_API_KEY "
            f"(a Cloudflare API token with the Workers AI permission).")

    body = {"model": config.LLAMA_MODEL, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "seed": config.LLAMA_SEED}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{api_base_url()}/chat/completions"

    wait = 2.0
    for attempt in range(config.LLAMA_MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            response = post(url, json=body, headers=headers, timeout=config.LLAMA_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            error = f"connection problem: {e}"
        else:
            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                return {"text": data["choices"][0]["message"]["content"].strip(),
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "seconds": round(time.perf_counter() - start, 2)}
            if response.status_code in (401, 403):
                raise LlamaError("The API key was rejected. Check LLAMA_API_KEY in .env.")
            if response.status_code == 402:
                raise LlamaError("The API account has no credit left. Add credit on the "
                                 "provider's website and run again.")
            if response.status_code == 429 and is_daily_limit(response.text):
                raise LlamaError("The free daily allowance is used up. Finished work is saved; "
                                 "run again tomorrow (the allowance resets at 00:00 UTC, "
                                 "5:30 am in Sri Lanka).")
            if response.status_code != 429 and response.status_code < 500:
                raise LlamaError(f"API error {response.status_code}: {response.text[:300]}")
            error = f"HTTP {response.status_code}"
            # The API often says how long to wait before trying again
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    wait = max(wait, float(retry_after))
                except ValueError:
                    pass
        if attempt < config.LLAMA_MAX_RETRIES:
            print(f"    {error}; waiting {wait:.0f} s and trying again ...")
            time.sleep(wait)
            wait = min(wait * 2, 120)
    raise LlamaError(f"Gave up after {config.LLAMA_MAX_RETRIES} retries ({error}).")
