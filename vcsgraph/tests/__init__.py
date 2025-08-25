# Copyright (C) 2025 Breezy Developers
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

import os
import shutil
import tempfile
import unittest
from unittest import TestCase

__all__ = ["TestCaseInTempDir"]


class TestCaseInTempDir(TestCase):
    """Minimal TestCase that runs in a temporary directory.

    Only implements the functionality actually needed by vcsgraph tests.
    """

    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix="vcsgraph_test_")
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir)
        super().tearDown()

    def assertPathExists(self, path):
        """Fail unless path exists."""
        self.assertTrue(os.path.lexists(path), f"{path} does not exist")

    def assertPathDoesNotExist(self, path):
        """Fail if path exists."""
        self.assertFalse(os.path.lexists(path), f"{path} exists")


def test_suite() -> unittest.TestSuite:
    names = [
        "known_graph",
        "graph",
        "tsort",
    ]
    module_names = ["vcsgraph.tests.test_" + name for name in names]
    loader = unittest.TestLoader()
    return loader.loadTestsFromNames(module_names)
