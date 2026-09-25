"""
normalize.py
Name and address normalization.

Critical constraint from the problem statement: training data is US/India,
test adds France, and the dataset also contains non-Latin scripts
(e.g. Devanagari business names). Normalization must NOT force everything
to ASCII — that would destroy signal for Indian/French records. We use
Unicode NFKC normalization (safe, script-preserving) rather than
transliteration or ASCII-folding.
"""
from __future__ import annotations

import re
import unicodedata

# Legal-suffix / abbreviation expansions. Keys must be applied on
# word-boundary tokens after lowercasing, so keep them lowercase.
LEGAL_SUFFIX_MAP = {
    "inc": "incorporated", "incorporated": "incorporated",
    "corp": "corporation", "corporation": "corporation",
    "co": "company", "company": "company",
    "ltd": "limited", "limited": "limited",
    "pvt": "private", "private": "private",
    "llc": "llc", "llp": "llp",
    "plc": "plc",
    "gmbh": "gmbh", "sarl": "sarl", "sas": "sas",  # France
}

ADDRESS_ABBR_MAP = {
    "rd": "road", "road": "road",
    "st": "street", "str": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "avenue": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard",
    "dr": "drive", "drive": "drive",
    "ln": "lane", "lane": "lane",
    "hwy": "highway", "highway": "highway",
    "apt": "apartment", "apartment": "apartment",
    "fl": "floor", "floor": "floor",
    "no": "number", "number": "number",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")


def _unicode_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    return text.strip()


def _basic_clean(text: str) -> str:
    text = _unicode_clean(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(name: str) -> str:
    text = _basic_clean(name)
    if not text:
        return text
    tokens = [LEGAL_SUFFIX_MAP.get(tok, tok) for tok in text.split(" ")]
    return " ".join(tokens)


def normalize_address(address: str) -> str:
    text = _basic_clean(address)
    if not text:
        return text
    tokens = [ADDRESS_ABBR_MAP.get(tok, tok) for tok in text.split(" ")]
    return " ".join(tokens)


def tokenize(text: str) -> list:
    if not text:
        return []
    return text.split(" ")


def extract_numeric_tokens(text: str) -> set:
    """House numbers, PIN/postal codes, etc. — any standalone digit run."""
    if not text:
        return set()
    return set(_DIGIT_RE.findall(text))


def char_ngrams(text: str, n: int = 3) -> set:
    text = text.replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def add_normalized_columns(df, name_col="business_name", addr_col="business_address"):
    """Adds norm_name, norm_address, name_tokens (space-joined), numeric_tokens columns in place."""
    df = df.copy()
    df["norm_name"] = df[name_col].map(normalize_name)
    df["norm_address"] = df[addr_col].map(normalize_address)
    df["numeric_tokens"] = df["norm_address"].map(lambda a: ",".join(sorted(extract_numeric_tokens(a))))
    return df
