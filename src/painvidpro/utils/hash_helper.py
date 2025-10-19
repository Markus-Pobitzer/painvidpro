import hashlib


def short_hash(text: str, length: int = 6) -> str:
    """Returns a shortened SHA-256 hash of the input text."""
    full_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return full_hash[:length]
