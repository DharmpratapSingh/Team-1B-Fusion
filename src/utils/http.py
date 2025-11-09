from typing import Any, Dict, Optional
import time
import requests

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3


def _request(
    method: str,
    url: str,
    *,
    timeout: Optional[float] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = 0.5,
    raise_for_status: bool = True,
    **kwargs: Any,
) -> requests.Response:
    timeout_seconds = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.request(method=method.upper(), url=url, timeout=timeout_seconds, **kwargs)
            if raise_for_status:
                resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001 - surfaced to caller after retries
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_base_seconds * (2 ** attempt))
                continue
            raise


def get(url: str, **kwargs: Any) -> requests.Response:
    return _request("GET", url, **kwargs)


def post(url: str, **kwargs: Any) -> requests.Response:
    return _request("POST", url, **kwargs)












