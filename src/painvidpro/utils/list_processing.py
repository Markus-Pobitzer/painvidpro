"""Utility function for list processing."""

from typing import Any, List


def batch_list(input_list: List[Any], batch_size: int):
    """Batches the list."""
    return [input_list[i : i + batch_size] for i in range(0, len(input_list), batch_size)]
