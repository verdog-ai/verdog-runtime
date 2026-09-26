"""The runtime distribution is usable without the management client."""

import re
from importlib import metadata

from verdog_runtime import locking, runs


def test_runtime_distribution_excludes_management_client() -> None:
    distribution = metadata.distribution("verdog-runtime")
    assert not any(
        entry.name == "verdog" for entry in distribution.entry_points
    )
    assert all(
        str(path) != "verdog_runtime/cli/main.py"
        and not str(path).startswith("verdog_cli/")
        for path in distribution.files or ()
    )
    assert all(
        not re.match(r"(?:verdog-cli|packaging|ty)(?:\W|$)", requirement)
        for requirement in distribution.requires or ()
    )
    assert runs.RunManifest.__module__ == "verdog_runtime._run_model"
    assert runs.RunStore.__module__ == "verdog_runtime._run_store"
    assert runs.LifecycleCommand.__module__ == "verdog_runtime._lifecycle"
    assert locking.locked_file.__module__ == "verdog_runtime._file_lock"
