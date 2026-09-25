import json
import sys
from pathlib import Path

import pytest

import verdog_runtime._checkpoint_compatibility as compatibility_module
from verdog_runtime._checkpoint_compatibility import (
    checkpoint_compatibility,
    compatibility_drift,
)


def test_source_fingerprint_changes_with_authored_code(tmp_path: Path) -> None:
    source = tmp_path / "src/example"
    source.mkdir(parents=True)
    authored = source / "impl.py"
    authored.write_text("VALUE = 1\n")
    first = checkpoint_compatibility(tmp_path)

    authored.write_text("VALUE = 2\n")
    second = checkpoint_compatibility(tmp_path)

    assert first["source_sha256"] != second["source_sha256"]
    assert compatibility_drift(first, second) == "changed source_sha256"


def test_fingerprint_records_nonempty_interpreter_identity(
    tmp_path: Path,
) -> None:
    fingerprint = checkpoint_compatibility(tmp_path)

    assert fingerprint["python_implementation"] == sys.implementation.name
    assert fingerprint["python_cache_tag"] == (
        sys.implementation.cache_tag or "unavailable"
    )
    assert fingerprint["platform"] == sys.platform
    assert fingerprint["format"] == "3"
    assert fingerprint["execution_model"] == "synchronous-activation-stack-v1"
    assert all(fingerprint.values())


def test_source_fingerprint_follows_symlinked_authored_code(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    authored = external / "impl.py"
    authored.write_text("VALUE = 1\n")
    source = tmp_path / "src/example"
    source.mkdir(parents=True)
    (source / "linked").symlink_to(external, target_is_directory=True)
    first = checkpoint_compatibility(tmp_path)

    authored.write_text("VALUE = 2\n")
    second = checkpoint_compatibility(tmp_path)

    assert first["source_sha256"] != second["source_sha256"]


def test_source_fingerprint_ignores_project_metadata(tmp_path: Path) -> None:
    source = tmp_path / "src/example"
    source.mkdir(parents=True)
    (source / "impl.py").write_text("VALUE = 1\n")
    project = tmp_path / "project.json"
    project.write_text('{"description": "before"}\n')
    first = checkpoint_compatibility(tmp_path)

    project.write_text('{"description": "after"}\n')
    second = checkpoint_compatibility(tmp_path)

    assert first["source_sha256"] == second["source_sha256"]


def test_source_fingerprint_includes_synced_source_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned = tmp_path / "project/src/example"
    owned.mkdir(parents=True)
    (owned / "impl.py").write_text("VALUE = 1\n")
    external = tmp_path / "external/src/dependency"
    external.mkdir(parents=True)
    dependency = external / "impl.py"
    dependency.write_text("VALUE = 1\n")
    marker = tmp_path / "environment/.verdog-environment.json"
    marker.parent.mkdir()
    marker.write_text(
        json.dumps(
            {
                "source_roots": [
                    str((tmp_path / "project/src").resolve()),
                    str((tmp_path / "external/src").resolve()),
                ]
            }
        )
    )
    monkeypatch.setattr(compatibility_module, "_ENVIRONMENT_MARKER", marker)
    first = checkpoint_compatibility(tmp_path / "project")

    dependency.write_text("VALUE = 2\n")
    second = checkpoint_compatibility(tmp_path / "project")

    assert first["source_sha256"] != second["source_sha256"]


def test_compatibility_reports_shape_drift() -> None:
    assert compatibility_drift({"a": "1", "b": "2"}, {"a": "2", "c": "3"}) == (
        "changed a; missing b; added c"
    )
