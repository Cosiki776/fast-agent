from __future__ import annotations

from typing import TypedDict


class InputRecord(TypedDict, total=False):
    identifier: str
    region: str


def process_records(records: list[InputRecord]) -> list[str]:
    """Format records for downstream batch delivery."""
    return [f"{record['identifier']}:{record['region']}" for record in records]
