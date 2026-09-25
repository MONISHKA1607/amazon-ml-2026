"""
normalize.py

Name and address normalization.

The pipeline keeps the original Unicode representation and additionally
creates transliterated representations. This allows cross-script records
such as English <-> Devanagari or English <-> Gujarati to share blocking
signals without destroying the original-script information.
"""

from __future__ import annotations

import re
import unicodedata

from anyascii import anyascii


# Legal-suffix / abbreviation expansions.
LEGAL_SUFFIX_MAP = {
    "inc": "incorporated",
    "incorporated": "incorporated",
    "corp": "corporation",
    "corporation": "corporation",
    "co": "company",
    "company": "company",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "gmbh": "gmbh",
    "sarl": "sarl",
    "sas": "sas",
}


ADDRESS_ABBR_MAP = {
    "rd": "road",
    "road": "road",
    "st": "street",
    "str": "street",
    "street": "street",
    "ave": "avenue",
    "av": "avenue",
    "avenue": "avenue",
    "blvd": "boulevard",
    "boulevard": "boulevard",
    "dr": "drive",
    "drive": "drive",
    "ln": "lane",
    "lane": "lane",
    "hwy": "highway",
    "highway": "highway",
    "apt": "apartment",
    "apartment": "apartment",
    "fl": "floor",
    "floor": "floor",
    "no": "number",
    "number": "number",
}


_PUNCT_RE = re.compile(
    r"[^\w\s]",
    flags=re.UNICODE,
)

_WS_RE = re.compile(
    r"\s+"
)

_DIGIT_RE = re.compile(
    r"\d+"
)


def _unicode_clean(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    return text.strip()


def _basic_clean(text: str) -> str:
    text = _unicode_clean(text)

    text = text.lower()

    text = text.replace(
        "&",
        " and ",
    )

    text = _PUNCT_RE.sub(
        " ",
        text,
    )

    text = _WS_RE.sub(
        " ",
        text,
    ).strip()

    return text


def normalize_name(name: str) -> str:
    """
    Preserve Unicode script while applying standard name normalization.
    """

    text = _basic_clean(name)

    if not text:
        return text

    tokens = [
        LEGAL_SUFFIX_MAP.get(
            tok,
            tok,
        )
        for tok in text.split(" ")
    ]

    return " ".join(tokens)


def normalize_address(address: str) -> str:
    """
    Preserve Unicode script while applying standard address normalization.
    """

    text = _basic_clean(address)

    if not text:
        return text

    tokens = [
        ADDRESS_ABBR_MAP.get(
            tok,
            tok,
        )
        for tok in text.split(" ")
    ]

    return " ".join(tokens)


def transliterate_text(text: str) -> str:
    """
    Convert Unicode text into a Latin-script approximation.

    This is an additional representation only. The original normalized
    Unicode representation is always preserved separately.
    """

    text = _unicode_clean(text)

    if not text:
        return ""

    transliterated = anyascii(text)

    transliterated = _basic_clean(
        transliterated
    )

    return transliterated


def transliterate_name(name: str) -> str:
    """
    Transliterate a raw business name and then apply the same legal-suffix
    normalization used by normalize_name().
    """

    text = transliterate_text(name)

    if not text:
        return ""

    tokens = [
        LEGAL_SUFFIX_MAP.get(
            tok,
            tok,
        )
        for tok in text.split(" ")
    ]

    return " ".join(tokens)


def transliterate_address(address: str) -> str:
    """
    Transliterate a raw business address and then apply the same address
    abbreviation normalization used by normalize_address().
    """

    text = transliterate_text(address)

    if not text:
        return ""

    tokens = [
        ADDRESS_ABBR_MAP.get(
            tok,
            tok,
        )
        for tok in text.split(" ")
    ]

    return " ".join(tokens)


def tokenize(text: str) -> list:
    if not text:
        return []

    return text.split(" ")


def extract_numeric_tokens(text: str) -> set:
    """
    House numbers, PIN/postal codes, etc.
    Any standalone digit run.
    """

    if not text:
        return set()

    return set(
        _DIGIT_RE.findall(text)
    )


def char_ngrams(
    text: str,
    n: int = 3,
) -> set:

    text = text.replace(
        " ",
        "",
    )

    if len(text) < n:
        return {text} if text else set()

    return {
        text[i:i + n]
        for i in range(
            len(text) - n + 1
        )
    }


def add_normalized_columns(
    df,
    name_col="business_name",
    addr_col="business_address",
):
    """
    Add normalized and transliterated representations.

    Existing columns remain unchanged:

        norm_name
        norm_address
        numeric_tokens

    New columns:

        translit_name
        translit_address
    """

    df = df.copy()

    # Original Unicode-preserving representations.
    df["norm_name"] = df[name_col].map(
        normalize_name
    )

    df["norm_address"] = df[addr_col].map(
        normalize_address
    )

    # Additional Latin-script representations.
    df["translit_name"] = df[name_col].map(
        transliterate_name
    )

    df["translit_address"] = df[addr_col].map(
        transliterate_address
    )

    # Numeric address evidence.
    df["numeric_tokens"] = df[
        "norm_address"
    ].map(
        lambda a: ",".join(
            sorted(
                extract_numeric_tokens(a)
            )
        )
    )

    return df