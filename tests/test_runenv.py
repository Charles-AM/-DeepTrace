"""Environment capture for repeatability claims."""

import json
from pathlib import Path

from src.runenv import ENV_KEYS, capture, git_state, hash_file


def test_hash_file_is_stable_and_content_sensitive(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_bytes(b"same"); b.write_bytes(b"same"); c.write_bytes(b"diff")
    assert hash_file(a) == hash_file(b) != hash_file(c)


def test_git_state_reports_commit_and_cleanliness():
    g = git_state(".")
    assert set(g) == {"commit", "dirty", "branch"}
    assert isinstance(g["dirty"], bool)


def test_capture_records_the_numerics_relevant_environment():
    rec = capture("t")
    assert set(rec["env"]) == set(ENV_KEYS)
    for k in ("tag", "platform", "python", "git", "torch"):
        assert k in rec


def test_capture_hashes_the_manifest_when_given_one(tmp_path):
    m = tmp_path / "m.csv"; m.write_text("path,label,split\n")
    rec = capture("t", manifest=m)
    assert rec["manifest"]["sha256"] == hash_file(m)


def test_capture_omits_manifest_block_when_the_file_is_absent(tmp_path):
    assert "manifest" not in capture("t", manifest=tmp_path / "nope.csv")


def test_two_captures_differ_only_where_expected():
    """The point of the file: a diff of two of these localises a discrepancy."""
    a, b = capture("a"), capture("b")
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"tag"}
