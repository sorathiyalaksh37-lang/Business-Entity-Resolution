"""
utils.py — Shared utility functions for Business Entity Resolution pipeline.

Phase 1: Normalization helpers used by ALL subsequent phases.
"""

import re
import unicodedata
from typing import Optional

# ---------------------------------------------------------------------------
# Abbreviation dictionaries
# ---------------------------------------------------------------------------

NAME_ABBREVS = {
    # Legal suffixes
    "pvt": "private",
    "priv": "private",
    "ltd": "limited",
    "llc": "llc",
    "llp": "llp",
    # NOTE: 'inc' is NOT expanded → EDA shows 'inc' appears 238K times but
    # 'incorporated' only 8 times. Expanding would BREAK matches, not fix them.
    # rapidfuzz token_set_ratio handles this in Phase 3 feature engineering.
    "corp": "corporation",
    "co": "company",
    "grp": "group",
    "intl": "international",
    "natl": "national",
    "mfg": "manufacturing",
    "mfr": "manufacturer",
    "svcs": "services",
    "svc": "service",
    "assoc": "associates",
    "assn": "association",
    "dept": "department",
    "bros": "brothers",
    "bro": "brother",
    "ent": "enterprises",
    "entrp": "enterprises",
    "enterp": "enterprises",
    # Conjunctions
    "&": "and",
    "+": "and",
}

ADDRESS_ABBREVS = {
    # Street types
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "sq": "square",
    "hwy": "highway",
    "pkwy": "parkway",
    "expy": "expressway",
    "fwy": "freeway",
    "trl": "trail",
    "xing": "crossing",
    "jct": "junction",
    # Directions
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    # Building types
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "dept": "department",
    # Indian address
    "nagar": "nagar",
    "marg": "marg",
    "chowk": "chowk",
    "colony": "colony",
}


def unicode_normalize(text: str) -> str:
    """Normalize unicode characters to ASCII equivalents where possible."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def normalize_name(name: Optional[str]) -> str:
    """
    Normalize a business name for blocking and feature computation.
    Steps: unicode → lowercase → expand abbreviations → remove punctuation → collapse whitespace.
    """
    if not name or not isinstance(name, str):
        return ""
    text = unicode_normalize(name)
    text = text.lower()
    # Replace punctuation with space (keep alphanumeric + space)
    text = re.sub(r"[^\w\s]", " ", text)
    # Tokenize and expand abbreviations
    tokens = text.split()
    tokens = [NAME_ABBREVS.get(t, t) for t in tokens]
    # Remove very short noise tokens (single chars that are not meaningful)
    tokens = [t for t in tokens if len(t) > 0]
    return " ".join(tokens).strip()


def normalize_address(address: Optional[str]) -> str:
    """
    Normalize a business address for blocking and feature computation.
    Steps: unicode → lowercase → expand abbreviations → remove punctuation → collapse whitespace.
    """
    if not address or not isinstance(address, str):
        return ""
    text = unicode_normalize(address)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    tokens = [ADDRESS_ABBREVS.get(t, t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 0]
    return " ".join(tokens).strip()


def extract_pin_zip(address: Optional[str]) -> Optional[str]:
    """Extract PIN code (India, 6 digits) or ZIP code (US, 5 digits) from address string."""
    if not address or not isinstance(address, str):
        return None
    # US ZIP: 5 digits (optionally followed by -4 digit extension)
    us_zip = re.search(r"\b(\d{5})(?:-\d{4})?\b", address)
    # India PIN: 6 digits
    in_pin = re.search(r"\b(\d{6})\b", address)
    if in_pin:
        return in_pin.group(1)
    if us_zip:
        return us_zip.group(1)
    return None


def extract_country_normalized(country: Optional[str]) -> str:
    """Normalize country string to a canonical form."""
    if not country or not isinstance(country, str):
        return "unknown"
    c = country.strip().lower()
    mapping = {
        "us": "us", "usa": "us", "united states": "us", "united states of america": "us",
        "india": "india", "in": "india",
        "france": "france", "fr": "france",
    }
    return mapping.get(c, c)


def get_name_tokens(normalized_name: str) -> set:
    """Return set of tokens from a normalized name (for Jaccard similarity)."""
    return set(normalized_name.split())


def get_address_tokens(normalized_address: str) -> set:
    """Return set of tokens from a normalized address."""
    return set(normalized_address.split())


def jaccard(set1: set, set2: set) -> float:
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


if __name__ == "__main__":
    # Quick sanity checks
    print(normalize_name("ABC Pvt. Ltd. & Associates"))
    # Expected: "abc private limited and associates"

    print(normalize_name("McDonald's Corp."))
    # Expected: "mcdonald s corporation"

    print(normalize_address("123 Main St, Apt 4B, New York, NY 10001"))
    # Expected: "123 main street apartment 4b new york ny 10001"

    print(extract_pin_zip("Near SBI ATM, Koramangala, Bangalore - 560034"))
    # Expected: "560034"

    print(extract_pin_zip("1234 Oak Blvd, Austin, TX 78701"))
    # Expected: "78701"

    print(extract_country_normalized("US"))
    # Expected: "us"
