# Verdog runtime

The Python runtime and command-line client for [Verdog](https://github.com/verdog-ai/verdog-website).
It provides workflow declarations, the interpreter, agent adapters, durable run resumption,
and the `verdog` command. Project generation and verification use the hosted backend;
this package contains no backend or compiler implementation.
Requires Python 3.12 or newer.

Install or update the command-line client with Python 3.12 or newer:

```sh
uv tool install --upgrade 'verdog-runtime>=0.1.1'
verdog --help
```

The client uses `https://157.180.79.112` by default. Use
`verdog --backend-origin URL ...` to select another backend. Saved terminal sessions
retain the backend used when signing in.

Generated projects also declare `verdog-runtime` as a dependency. For local development,
install this checkout with `uv pip install .`.

## Development

```sh
uv sync --locked --no-editable
uv run --no-sync pytest
uv run --no-sync pyright
uv build
```

The tests and type-checking configuration are self-contained in this repository.
Use a regular installation (`--no-editable`) because `verdog sync` copies the installed
distribution into workflow environments.

See the [workflow declarations](verdog_runtime/declarations/README.md),
[interpreter](verdog_runtime/interpreter/README.md), and
[user documentation](https://github.com/verdog-ai/verdog-website/tree/main/site).

## Release

[release.yml](.github/workflows/release.yml) runs when a `v*` tag is pushed,
like `v0.1.1`. It checks that the tag matches `project.version`, runs tests and
type checks on Python 3.12, builds the wheel and source distribution, checks
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
git push origin main
git tag v0.1.1
git push origin v0.1.1
```

Use a new version and matching tag for each subsequent release.

## License

Verdog runtime and its command-line client are licensed under the GNU Affero General Public License,
version 3 only (`AGPL-3.0-only`). See [LICENSE](LICENSE).
