# Runtime development

## Development

```sh
uv sync --locked --no-editable
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyright
uv run --no-sync pytest
uv build
```

The tests and type-checking configuration are self-contained in this repository.
Use a regular installation (`--no-editable`) because `verdog sync` copies the installed
distribution into workflow environments.

### Code quality

Follow the [Google Python style guide](https://google.github.io/styleguide/pyguide.html).
Ruff enforces 80-column formatting, absolute imports, import sorting,
Google-style docstrings, and common correctness and simplification checks.
Import modules rather than their members, except for typing names and deliberate
public re-exports. Keep error messages, checkpoint formats, and public APIs stable
when refactoring. Documentation should explain contracts and non-obvious constraints.

Run `uv run --no-sync ruff format .` to format changes. Both pull requests and
releases check formatting, lint, types, and tests. The existing strict Pyright
configuration remains the authority for types; no type errors are suppressed to
satisfy a formatter.

For a deeper review, use the pinned analysis tools:

```sh
uv run --no-sync pylint verdog_runtime --reports=no
uv run --no-sync radon cc verdog_runtime -s -n C
uv run --no-sync radon mi verdog_runtime -s
```

Pylint and Radon are review aids, not score targets. Prioritize functions that
combine validation, mutation, and recovery; split them at those boundaries rather
than adding helpers just to meet a numeric threshold. Keep rollback and trust-boundary
checks intact. Cyclomatic complexity counts independent control-flow paths; inspect
both the largest function and any helpers extracted from it.

Compatibility exceptions are narrow: public package facades retain re-exports;
`TypeVar` and type-alias syntax remains where runtime introspection and checkpoint
compatibility depend on it; dynamic dataclass access uses `getattr` explicitly.
Tests use descriptive names and assertions instead of mandatory API docstrings.

## Run monitoring

`verdog_runtime.runs.load_run_header(output_dir)` returns a validated `RunHeader`
with run identity, launch information, and the status recorded in `run.json`.
It does not enumerate checkpoints, load their manifests, or inspect artifacts.
Its `updated_at` is the recorded run timestamp; checkpoint activity is reported
by `trace.log` independently. `RunHeader.as_summary()` omits checkpoint and
session summaries.

Use `run_is_active(output_dir)` to check the existing runtime lease. The operating
system releases that lock when the process exits, including forced termination;
a recorded `running` status without the lease therefore indicates interruption.
Monitoring clients must check `header.project_root` against their project before
displaying a run. Full `RunStore` reads retain checkpoint validation for explicit
inspection, resume, and fork.

Checkpoint storage schema 4 writes a full artifact inventory at the first
artifact-bearing boundary and add-only deltas thereafter. Each delta names the
previous artifact-bearing checkpoint, including across unavailable boundaries.
Readers retain shared inventory layers and reconstruct the selected boundary
when validating or materializing artifacts. Checkpointed files and directory
modes remain immutable; changes and deletions are integrity errors.

Existing schema 3 checkpoints with full inventories remain readable and can be
followed by new delta checkpoints. Existing files are not rewritten. Child
processes still send a complete inventory captured at their execution boundary;
the parent computes its stored delta without rescanning the child's output.
Run-history summaries and the child protocol retain their existing versions.

## Release

[release.yml](.github/workflows/release.yml) runs when a `v*` tag is pushed.
It checks that the tag matches `project.version`, runs style, tests, and type checks
on Python 3.12, builds the wheel and source distribution, checks
the installed wheel, and publishes both distributions to PyPI.

Configure this once before the first release:

1. Create the GitHub environment `pypi` in this repository's settings.
2. Add a [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/):
   project `verdog-runtime`, owner `verdog-ai`, repository `verdog-runtime`,
   workflow filename `release.yml`, environment `pypi`.
   For a new PyPI project, add it as a pending publisher under your account's
   Publishing settings. For an existing project, use its Publishing settings.
   No API-token secret is needed.

Commit and push the repository contents, including this workflow. For each
release, update `project.version` in `pyproject.toml`, run `uv lock`, and commit
those changes. Then push the matching tag:

```sh
release_version=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
git push origin main
git tag "v$release_version"
git push origin "v$release_version"
```

Use a new version and matching tag for each subsequent release.
