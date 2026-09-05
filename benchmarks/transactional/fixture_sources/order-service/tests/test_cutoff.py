from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from order_service import can_create_order


class OrderCutoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cutoff = datetime(2026, 8, 24, 16, 0, tzinfo=UTC)

    def test_order_before_cutoff_is_allowed(self) -> None:
        self.assertTrue(can_create_order(self.cutoff - timedelta(seconds=1), self.cutoff))

    def test_order_at_cutoff_is_allowed(self) -> None:
        self.assertTrue(can_create_order(self.cutoff, self.cutoff))

    def test_order_after_cutoff_is_rejected(self) -> None:
        self.assertFalse(can_create_order(self.cutoff + timedelta(seconds=1), self.cutoff))


if __name__ == "__main__":
    unittest.main()
