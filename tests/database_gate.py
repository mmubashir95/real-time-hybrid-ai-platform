"""Gate PostgreSQL-backed tests so a missing database is never a silent pass.

By default these tests skip when TEST_DATABASE_URL is unset, which keeps the
offline suite fast. Setting REQUIRE_DATABASE_TESTS=1 (for example in CI or
before a release) turns that skip into a hard failure, so "54 skipped" cannot
hide an unconfigured database.
"""

from __future__ import annotations

import os
from typing import Callable, TypeVar
import unittest


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")
_TestClass = TypeVar("_TestClass", bound=type)


def requires_test_database() -> Callable[[_TestClass], _TestClass]:
    """Return a class decorator that skips, or fails when the database is required."""
    if TEST_DATABASE_URL:
        return lambda test_class: test_class
    if os.getenv("REQUIRE_DATABASE_TESTS") == "1":
        raise RuntimeError(
            "REQUIRE_DATABASE_TESTS=1 but TEST_DATABASE_URL is not set"
        )
    return unittest.skip(
        "TEST_DATABASE_URL is not set; set it (and REQUIRE_DATABASE_TESTS=1 "
        "to forbid skipping) to run PostgreSQL tests"
    )
