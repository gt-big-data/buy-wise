
import re
from urllib.parse import urlparse


_ASIN_RE = re.compile(r'^[A-Z0-9]{10}$')


_AMAZON_DOMAINS = {
    "amazon.com",
    "amazon.co.uk",
    "amazon.co.jp",
    "amazon.ca",
    "amazon.com.au",
    "amazon.com.br",
    "amazon.com.mx",
    "amazon.com.sg",
    "amazon.com.tr",
    "amazon.de",
    "amazon.es",
    "amazon.fr",
    "amazon.in",
    "amazon.it",
    "amazon.nl",
    "amazon.pl",
    "amazon.sa",
    "amazon.se",
    "amazon.ae",
    "amazon.eg",
}

_PATH_PATTERNS = [
    re.compile(r'^/(?:[^/]+/)?dp/([A-Z0-9]{10})(?:/|$)', re.IGNORECASE),
    re.compile(r'^/gp/product/([A-Z0-9]{10})(?:/|$)', re.IGNORECASE),
    re.compile(r'^/exec/obidos/ASIN/([A-Z0-9]{10})(?:/|$)', re.IGNORECASE),
    re.compile(r'^/o/ASIN/([A-Z0-9]{10})(?:/|$)', re.IGNORECASE),
]



def extract_asin(url: str) -> str | None:
    if not isinstance(url, str):
        return None

    url = url.strip()

    try:
        parsed = urlparse(url)
    except Exception:
        return None


    if parsed.scheme.lower() not in ("http", "https"):
        return None
    hostname = (parsed.hostname or "").lower()
    if not _is_amazon_hostname(hostname):
        return None
    path = parsed.path  
    asin = _asin_from_path(path)
    if asin is None:
        return None

    return asin.upper() if _ASIN_RE.match(asin.upper()) else None


def _is_amazon_hostname(hostname: str) -> bool:
    for domain in _AMAZON_DOMAINS:
        if hostname == domain or hostname.endswith("." + domain):
            return True
    return False


def _asin_from_path(path: str) -> str | None:
    """Try every known path pattern and return the first ASIN match."""
    for pattern in _PATH_PATTERNS:
        m = pattern.search(path)
        if m:
            return m.group(1)
    return None
