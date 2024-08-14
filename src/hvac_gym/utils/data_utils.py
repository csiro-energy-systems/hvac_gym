# The Software is copyright (c) CSIRO ABN 41 687 119 230
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def unique(seq: Sequence[T]) -> list[T]:
    """Fast, order preserving list uniquification.
    See https://www.peterbe.com/plog/fastest-way-to-uniquify-a-list-in-python-3.6
    """
    return list(dict.fromkeys(seq))
