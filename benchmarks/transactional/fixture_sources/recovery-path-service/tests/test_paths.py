from __future__ import annotations

import unittest

from workspace_utils import is_within_workspace


class WorkspacePathTests(unittest.TestCase):
    def test_normal_child_is_allowed(self) -> None:
        self.assertTrue(is_within_workspace("/tmp/project", "/tmp/project/src/app.py"))

    def test_root_itself_is_allowed(self) -> None:
        self.assertTrue(is_within_workspace("/tmp/project", "/tmp/project"))

    def test_similar_prefix_sibling_is_rejected(self) -> None:
        self.assertFalse(is_within_workspace("/tmp/project", "/tmp/project2/secret.txt"))

    def test_parent_escape_is_rejected(self) -> None:
        self.assertFalse(is_within_workspace("/tmp/project", "/tmp/project/../secret.txt"))

    def test_normalized_sibling_escape_is_rejected(self) -> None:
        self.assertFalse(is_within_workspace("/tmp/project", "/tmp/project/src/../../secret"))


if __name__ == "__main__":
    unittest.main()
