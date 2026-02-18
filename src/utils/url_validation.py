"""SSRF-safe URL validation and fetching."""

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_REDIRECTS = 5
BLOCKED_HOSTNAMES = {"metadata.google.internal"}
ALLOWED_SCHEMES = {"http", "https"}


class SSRFError(ValueError):
    """Raised when a URL targets a non-public address."""


def validate_url(url: str) -> str:
    """Validate that a URL targets a public internet host.

    Returns the URL unchanged if valid, raises SSRFError otherwise.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise SSRFError(f"Scheme {parsed.scheme!r} not allowed (must be http or https)")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL has no hostname")

    if hostname in BLOCKED_HOSTNAMES:
        raise SSRFError(f"Hostname {hostname!r} is blocked")

    try:
        addrinfo = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise SSRFError(f"Cannot resolve hostname {hostname!r}: {exc}") from exc

    for family, _, _, _, sockaddr in addrinfo:
        ip_str = sockaddr[0]
        ip = ipaddress.ip_address(ip_str)
        if not ip.is_global:
            raise SSRFError(
                f"Hostname {hostname!r} resolves to non-public IP {ip_str}"
            )

    return url


async def safe_fetch(url: str, timeout: float = 30.0) -> str:
    """Fetch a URL with SSRF protection, validating every redirect."""
    validate_url(url)

    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
        for _ in range(MAX_REDIRECTS):
            response = await client.get(url)
            if response.is_redirect:
                url = str(response.next_request.url)
                validate_url(url)
                continue
            response.raise_for_status()
            return response.text

    raise SSRFError(f"Too many redirects (>{MAX_REDIRECTS})")
