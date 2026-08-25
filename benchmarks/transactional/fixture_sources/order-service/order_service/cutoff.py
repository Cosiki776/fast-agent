from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


def can_create_order(created_at: datetime, cutoff: datetime) -> bool:
    """Return whether an order may be created before the daily cutoff."""
    return created_at < cutoff
