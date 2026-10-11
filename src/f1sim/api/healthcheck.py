"""Container health check for the API: ``python -m f1sim.api.healthcheck``.

Exits 0 when ``/health`` answers within two seconds and 1 otherwise, so Docker and Compose
can call one short command instead of an inline Python one-liner.
"""

import sys
import urllib.error
import urllib.request
from http import HTTPStatus

HEALTH_URL = "http://127.0.0.1:8000/health"
TIMEOUT_S = 2.0


def check(url: str = HEALTH_URL, timeout_s: float = TIMEOUT_S) -> bool:
    """Ask the API whether it is healthy.

    Args:
        url: Health endpoint to call.
        timeout_s: How long to wait for an answer.

    Returns:
        True if the endpoint answered with a 2xx status in time.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return HTTPStatus.OK <= response.status < HTTPStatus.MULTIPLE_CHOICES
    except (urllib.error.URLError, TimeoutError):
        return False


def main() -> int:
    """Run the health check.

    Returns:
        0 when healthy, 1 otherwise.
    """
    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
