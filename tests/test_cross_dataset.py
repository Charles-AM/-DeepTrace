"""Validation for cross-dataset target parsing (Task 7 / 13)."""

import pytest

from src.cross_dataset import _parse_targets


def test_parse_single_target():
    assert _parse_targets(["celebdf=/data/celeb"]) == {"celebdf": "/data/celeb"}


def test_parse_multiple_targets():
    got = _parse_targets(["celebdf=/a", "dfdc=/b/c", "df40=/d"])
    assert got == {"celebdf": "/a", "dfdc": "/b/c", "df40": "/d"}


def test_parse_path_with_equals_sign():
    assert _parse_targets(["x=/a=b"]) == {"x": "/a=b"}


def test_rejects_bare_spec():
    with pytest.raises(ValueError):
        _parse_targets(["/just/a/path"])
