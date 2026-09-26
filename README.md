# Verdog runtime

The Python workflow runtime for [Verdog](https://github.com/verdog-ai/verdog-website).
It provides workflow declarations, the interpreter, agent adapters, durable run
resumption, and typed workflow argument parsing. Requires Python 3.12 or newer.

Generated projects declare `verdog-runtime` as a dependency. Install it directly
for programmatic use:

```sh
uv pip install 'verdog-runtime>=0.1.3'
```

The separate [verdog-cli](https://github.com/verdog-ai/verdog-cli) package provides
the `verdog` command and talks to the hosted backend for generation and verification:

```sh
uv tool install --upgrade 'verdog-cli>=0.1.1'
verdog --help
```

The CLI depends on this runtime; the runtime does not depend on the CLI.
Workflow environments contain the runtime and workflow dependencies. Run history
and lifecycle transport are exposed through `verdog_runtime.runs`; advisory file
locks through `verdog_runtime.locking`; typed workflow argument parsing remains
available from `verdog_runtime.cli`.

The package split preserves run-history formats. Exact resume and fork still
check runtime and environment fingerprints, so checkpoints created before the
split require their original compatible runtime environment.

Runtime 0.1.3 removes the `web_search` invoker keyword; the updated backend also
removes the profile option.
Remove `options.web_search` from agent profiles in `project.json`; configure search
through provider-native `extra_args` instead: Codex uses `-c` and
`web_search="live"`; Claude uses `--tools` and `Read,Glob,Grep,WebSearch,WebFetch`.
Regenerate sources with an updated backend using `verdog generate` before using
runtime 0.1.3, including sources that previously passed `web_search=False`.

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

The September 2026 pass was measured against commit `e457564`, using the same
Ruff settings for both versions:

| Measure | Before | After |
| --- | ---: | ---: |
| Ruff diagnostics | 2,323 | 0 |
| Highest function cyclomatic complexity | 107 | 32 |
| Functions above complexity 20 | 11 | 9 |
| Mean function cyclomatic complexity | 4.64 | 4.56 |
| Function and nested-function count | 767 | 780 |

The largest changes separate dependency-removal validation from graph/file edits,
projection staging from rollback, and checkpoint validation from resource setup.
The extra functions represent those responsibilities; they do not change the
checkpoint or wire formats. Physical line counts rise with 80-column wrapping,
qualified names, and documentation, so line count alone is not the improvement
criterion.

See the [workflow declarations](verdog_runtime/declarations/README.md),
[interpreter](verdog_runtime/interpreter/README.md), and
[user documentation](https://github.com/verdog-ai/verdog-website/tree/main/site).

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

## License

Verdog runtime is licensed under the GNU Affero General Public License,
version 3 only (`AGPL-3.0-only`). See [LICENSE](LICENSE).
