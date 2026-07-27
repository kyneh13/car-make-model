from __future__ import annotations

import json
import os
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from openai import OpenAI


MODEL_NAME = os.getenv(
    "OPENAI_PRICE_MODEL",
    "gpt-5-mini",
)

CACHE_PATH = Path(
    "/workspace/cache/openai_car_prices.json"
)

CACHE_MAX_AGE = timedelta(days=30)
CACHE_LOCK = threading.Lock()


def unavailable(
    car_name: str,
    message: str,
) -> dict[str, Any]:
    return {
        "available": False,
        "car": car_name,
        "original_msrp": None,
        "current_low": None,
        "current_high": None,
        "current_midpoint": None,
        "trend": "unknown",
        "change_amount": None,
        "change_percent": None,
        "annual_change_percent": None,
        "summary": "",
        "method": (
            f"OpenAI {MODEL_NAME} with web search"
        ),
        "checked_date": date.today().isoformat(),
        "sources": [],
        "notice": message,
    }


def load_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}

    try:
        data = json.loads(
            CACHE_PATH.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, dict):
            return data

    except (OSError, json.JSONDecodeError):
        pass

    return {}


def save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = CACHE_PATH.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(
            cache,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(CACHE_PATH)


def cache_is_fresh(
    item: dict[str, Any],
) -> bool:
    checked_at = item.get("checked_at")

    if not checked_at:
        return False

    try:
        checked_time = datetime.fromisoformat(
            checked_at
        )
    except (TypeError, ValueError):
        return False

    return (
        datetime.now() - checked_time
        <= CACHE_MAX_AGE
    )


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    ).strip()

    start = cleaned.find("{")

    if start == -1:
        raise ValueError(
            "OpenAI response contained no JSON."
        )

    decoder = json.JSONDecoder()

    value, _ = decoder.raw_decode(
        cleaned[start:]
    )

    if not isinstance(value, dict):
        raise ValueError(
            "OpenAI response was not a JSON object."
        )

    return value


def number_or_none(
    value: Any,
) -> float | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        number = float(value)

    elif isinstance(value, str):
        cleaned = (
            value.replace("$", "")
            .replace(",", "")
            .replace("%", "")
            .strip()
        )

        try:
            number = float(cleaned)
        except ValueError:
            return None

    else:
        return None

    if number <= 0:
        return None

    return number


def find_model_year(
    car_name: str,
) -> int | None:
    match = re.search(
        r"\b(19[8-9][0-9]|20[0-2][0-9])\b",
        car_name,
    )

    if match:
        return int(match.group(1))

    return None


def extract_sources(
    response: Any,
) -> list[dict[str, str]]:
    try:
        payload = response.model_dump()
    except Exception:
        return []

    sources: list[dict[str, str]] = []
    found_urls: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            url = value.get("url")

            if (
                isinstance(url, str)
                and url.startswith(("http://", "https://"))
                and url not in found_urls
            ):
                title = value.get("title")

                if not isinstance(title, str):
                    title = "Pricing source"

                found_urls.add(url)

                sources.append({
                    "title": title.strip()
                    or "Pricing source",
                    "url": url,
                })

            for child in value.values():
                visit(child)

        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)

    return sources[:6]


def get_price_estimate(
    car_name: str,
) -> dict[str, Any]:
    cache_key = car_name.strip().lower()

    # Prevent several browser requests from
    # researching prices at the same time.
    with CACHE_LOCK:
        cache = load_cache()
        cached = cache.get(cache_key)

        if (
            isinstance(cached, dict)
            and cache_is_fresh(cached)
        ):
            print(
                f"Using cached OpenAI pricing: "
                f"{car_name}"
            )

            return cached

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            return unavailable(
                car_name,
                "OPENAI_API_KEY is not loaded.",
            )

        print(
            f"OpenAI is researching pricing: "
            f"{car_name}"
        )

        prompt = f"""
Use web search to research United States pricing
for this vehicle:

{car_name}

Current date: {date.today().isoformat()}

Find:

1. Approximate original BASE MSRP when new.
2. Current typical dealer used-retail low price.
3. Current typical dealer used-retail high price.
4. A short explanation of the estimate.

Prefer credible automotive sources such as
manufacturers, Edmunds, Kelley Blue Book,
JD Power, Cars.com, CarGurus, Autotrader,
and CARFAX.

Important rules:

- The label may not include trim, mileage,
  drivetrain, condition, options, or location.
- Use a broad national estimate.
- Exclude monthly payments and lease payments.
- Exclude auction, salvage, wholesale, and
  private-party prices.
- Do not invent missing information.
- Use null when a value cannot be verified.
- Dollar values must be plain numbers.
- Do not put citation symbols inside the JSON.
- Return only one valid JSON object.

Required output:

{{
  "original_msrp": number or null,
  "current_retail_low": number or null,
  "current_retail_high": number or null,
  "summary": "one short sentence"
}}
"""

        try:
            client = OpenAI(
                api_key=api_key,
                timeout=60.0,
                max_retries=2,
            )

            response = client.responses.create(
                model=MODEL_NAME,

                reasoning={
                    "effort": "low",
                },

                tools=[
                    {
                        "type": "web_search",
                        "search_context_size": "low",
                    }
                ],

                # Make sure a live web search occurs.
                tool_choice="required",

                include=[
                    "web_search_call.action.sources"
                ],

                # Force the final response to follow
                # this exact JSON structure.
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "car_price_estimate",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "original_msrp": {
                                    "type": [
                                        "number",
                                        "null",
                                    ]
                                },
                                "current_retail_low": {
                                    "type": [
                                        "number",
                                        "null",
                                    ]
                                },
                                "current_retail_high": {
                                    "type": [
                                        "number",
                                        "null",
                                    ]
                                },
                                "summary": {
                                    "type": "string"
                                },
                            },
                            "required": [
                                "original_msrp",
                                "current_retail_low",
                                "current_retail_high",
                                "summary",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },

                max_output_tokens=1200,
                store=False,
                input=prompt,
            )

            if response.status != "completed":
                raise ValueError(
                    "OpenAI response did not complete. "
                    f"Status: {response.status}; "
                    f"details: "
                    f"{response.incomplete_details}"
                )

            if not response.output_text.strip():
                raise ValueError(
                    "OpenAI returned empty "
                    "structured output."
                )

            data = json.loads(
                response.output_text
            )

            original_msrp = number_or_none(
                data.get("original_msrp")
            )

            current_low = number_or_none(
                data.get("current_retail_low")
            )

            current_high = number_or_none(
                data.get("current_retail_high")
            )

            # Reject clearly unrealistic dollar values.
            if (
                original_msrp is not None
                and not 5_000 <= original_msrp <= 1_000_000
            ):
                original_msrp = None

            if (
                current_low is not None
                and not 1_000 <= current_low <= 1_000_000
            ):
                current_low = None

            if (
                current_high is not None
                and not 1_000 <= current_high <= 1_000_000
            ):
                current_high = None

            if (
                current_low is not None
                and current_high is not None
                and current_low > current_high
            ):
                current_low, current_high = (
                    current_high,
                    current_low,
                )

            current_midpoint = None
            trend = "unknown"
            change_amount = None
            change_percent = None
            annual_change_percent = None

            if (
                current_low is not None
                and current_high is not None
            ):
                current_midpoint = (
                    current_low + current_high
                ) / 2

            if (
                original_msrp is not None
                and original_msrp > 0
                and current_midpoint is not None
            ):
                change_amount = (
                    current_midpoint
                    - original_msrp
                )

                change_percent = (
                    change_amount
                    / original_msrp
                    * 100
                )

                if change_percent > 2:
                    trend = "increased"

                elif change_percent < -2:
                    trend = "decreased"

                else:
                    trend = "about the same"

                model_year = find_model_year(
                    car_name
                )

                if model_year is not None:
                    age = max(
                        1,
                        date.today().year
                        - model_year,
                    )

                    annual_change_percent = (
                        (
                            (
                                current_midpoint
                                / original_msrp
                            )
                            ** (1 / age)
                        )
                        - 1
                    ) * 100

            result = {
                "available": any(
                    value is not None
                    for value in (
                        original_msrp,
                        current_low,
                        current_high,
                    )
                ),
                "car": car_name,
                "original_msrp": original_msrp,
                "current_low": current_low,
                "current_high": current_high,
                "current_midpoint":
                    current_midpoint,
                "trend": trend,
                "change_amount": change_amount,
                "change_percent": change_percent,
                "annual_change_percent":
                    annual_change_percent,
                "summary": str(
                    data.get("summary", "")
                ).strip(),
                "method": (
                    f"OpenAI {MODEL_NAME} "
                    f"with web search"
                ),
                "checked_date":
                    date.today().isoformat(),
                "checked_at":
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
                "sources": extract_sources(
                    response
                ),
                "notice": (
                    "Approximate U.S. retail estimate. "
                    "Actual value depends on trim, "
                    "mileage, condition, options, "
                    "drivetrain, and location."
                ),
            }

            cache[cache_key] = result
            save_cache(cache)

            return result

        except Exception as error:
            print(
                f"OpenAI pricing error: {error}"
            )

            return unavailable(
                car_name,
                f"Pricing lookup failed: {error}",
            )
