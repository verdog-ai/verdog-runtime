# Ownership

This file is maintained and reviewed by the user. Agents MUST NOT edit, delete,
rename, replace or regenerate it. Report outdated or conflicting instructions
and propose amendments in the conversation.

# Repository Scope

- This repository provides the public `verdog-runtime` distribution.
- It owns workflow declarations, typed argument parsing, execution, agent adapters,
  cancellation, process isolation, run storage, checkpoints, and lifecycle operations.
- It does not own the `verdog` application command, project generation, catalogue,
  backend authentication, or environment provisioning.
- Use [pyproject.toml](pyproject.toml) for package metadata, dependencies, and tooling;
  use [README.md](README.md) and the declarations/interpreter READMEs for runtime contracts.

# Dependency Chain

- Package dependencies are the third-party runtime libraries declared in the manifest.
  There is no CLI, compiler, service, website, or editor-extension package dependency.
- Consumers: generated workflow projects import the runtime; `verdog-cli` depends on
  it and provisions its installed distribution into workflow environments.
- The running example is a generated-project consumer. Its application dependencies
  belong to its workflow requirements, not to this runtime's dependency list.
- Backend compiler tests depend on the runtime to validate generated declarations.
  That development relationship is not a production backend/runtime package dependency.
- The compiler's manifest owns its generated-project runtime compatibility requirement;
  consumers must not infer that requirement from whichever runtime is locally installed.
- The extension consumes runtime behavior through the CLI. Its development integration
  checks also inspect runtime run-history contracts; coordinate those protocol changes.
- Agent adapters invoke configured external agent tools; preserve their execution and
  artifact contracts without coupling the runtime to editor or backend implementations.

# Design Constraints

- Never import CLI, compiler, or hosted-backend packages into the runtime.
  `verdog_runtime.cli` is typed workflow argument parsing, not the application CLI.
- Keep the package usable independently of the Verdog service and editor extension.
  Do not introduce catalogue authentication, a custom GitHub App, or OAuth registration.
- Preserve workflow/process isolation and the distinct in-process subroutine model.
  A child workflow's dependencies must not enter its caller's interpreter.
- Preserve declared identities, payload/configuration contracts, and generated-module
  compatibility; consult the public declarations before changing execution internals.
- Treat run-store schemas, checkpoint compatibility, continuation semantics, and child
  protocols as consumer contracts, not private formats that may change silently.
- Checkpoint restore must retain its compatibility and integrity checks. Preserve
  commit boundaries, failure recovery, cancellation, process cleanup, and file locking.
- Keep source/artifact path validation and containment checks at their trust boundaries.
  Do not weaken them to simplify implementation or satisfy an unrelated fixture.
- Agent execution and deserialization are execution paths, not source-only inspection.
  Runtime execution must not be used to inspect an untrusted catalogue release.
- Keep declared dependencies authoritative and avoid pulling authoring-only tools into
  workflow environments. The CLI owns installation and environment freshness checks.

# Development and Validation

- Follow the README's setup and the checks in
  [quality.yml](.github/workflows/quality.yml); use the existing Python toolchain.
- Before starting a local build, check whether another local build is active.
  Do not overlap local builds, and use at most 12 workers for a build.
- Use the manifest's quality configuration. After setup, run
  `uv run --no-sync ruff check .`, `uv run --no-sync ruff format --check .`,
  `uv run --no-sync pyright` and `uv run --no-sync pytest`.
  Run focused meaningful regressions during development; check distributions with
  `uv build` when packaging changes.
- Exercise failure, cancellation, child-process, and checkpoint boundaries when changes
  touch them; do not replace these checks with tests that merely repeat implementation.
- Validate public declaration or protocol changes against applicable CLI/backend/editor
  consumer checks. Ordinary runtime tests and builds remain independent of sibling repos.
- Preserve installed-package boundary checks: a runtime wheel must not provide the
  `verdog` executable or accidentally contain CLI/backend/compiler modules.

# Release Flow

- Commit/push authorization does not authorize release, publishing, tagging, or deployment;
  obtain explicit authorization for those actions.
- [release.yml](.github/workflows/release.yml) is the release source of truth:
  matching version tag, quality checks, distributions, installed-wheel boundary checks,
  then PyPI Trusted Publishing.
- Publish required runtime functionality before CLI or generated-project consumers
  declare a dependency on it. Dependencies must be available before their consumers.
- Coordinate generated-code compatibility with the compiler's declared requirement and
  shared run-history/checkpoint contracts with their consumers; a routine patch does not
  itself require raising every consumer's minimum runtime requirement.
