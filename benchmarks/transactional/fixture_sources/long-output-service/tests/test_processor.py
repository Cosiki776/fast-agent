from __future__ import annotations

import unittest

from batch_service import process_records


class BatchProcessorTests(unittest.TestCase):
    def test_explicit_region_is_preserved(self) -> None:
        self.assertEqual(
            process_records([{"identifier": "order-1", "region": "eu"}]),
            ["order-1:eu"],
        )

    def test_missing_region_uses_unknown(self) -> None:
        for index in range(5_000):
            print(f"processing record {index:04d}: awaiting normalized region")
        self.assertEqual(
            process_records([{"identifier": "order-2"}]),
            ["order-2:unknown"],
        )

    def test_empty_batch_remains_empty(self) -> None:
        self.assertEqual(process_records([]), [])


if __name__ == "__main__":
    unittest.main()
