# Contributing

Contributions are welcome. This page covers the local workflow; the same checks
run in [CI](https://github.com/wickeddoc/syncalong/actions).

## Setup

```bash
git clone https://github.com/wickeddoc/syncalong.git
cd syncalong
pip install -e ".[dev,docs]"
```

You also need **ffmpeg** on your `PATH` for any end-to-end run (the test suite
itself does not need it — see below).

## Quality gate

Four checks must pass before a change is merged. They mirror the CI workflow:

```bash
pytest                    # tests
ruff check .              # lint (includes import sorting + docstring rules)
black --check .           # formatting
pyright src               # type checking of the public surface
```

Autofix formatting and the fixable lint issues with:

```bash
black .
ruff check --fix .
```

### Tests

Tests live in `tests/test_core.py` and run in well under a second. They exercise
the whole text pipeline (parsing, scoring, DP alignment, formatting) and the
library API using synthetic `WordTimestamp` data and an **injected fake Whisper
model** — so no audio files, no model download, and no ffmpeg are required.

When adding behavior, add a test alongside the existing classes (for example
`TestDPAlign`, `TestAlignFacade`, `TestLRCFormatter`).

### Type checking

The package ships a `py.typed` marker, so the public API is type-checked.
`pyright src` must stay clean; keep annotations correct on anything exported
from `syncalong/__init__.py`.

### Docstrings

Every public class and function uses **Google-style** docstrings
(`Args:` / `Returns:` / `Raises:` / `Attributes:`). Ruff's pydocstyle rules
enforce their presence with the `google` convention. The API reference on this
site is generated from these docstrings by
[mkdocstrings](https://mkdocstrings.github.io/), so a good docstring *is* the
documentation.

## Text normalization

`textnorm.normalize()` is the single source of truth shared by the lyrics parser
and the transcriber. **Never reimplement normalization locally** — both sides of
the aligner must normalize identically or fuzzy matching silently degrades.

## Building the docs

```bash
mkdocs serve      # live-reload preview at http://127.0.0.1:8000
mkdocs build      # render the static site into ./site
```

## Documentation versions

The docs on [ReadTheDocs](https://syncalong.readthedocs.io/) build two moving
versions:

- **`stable`** — the highest semver tag (the current release). It's the
  **default** version, so the bare docs URL lands here — the docs that match
  what `pip install syncalong` installs.
- **`latest`** — `master`, i.e. **unreleased** work; it can show features that
  aren't on PyPI yet.

Two rules keep this honest:

1. **Releases promote themselves.** Tagging (e.g. `v2.0.0`, see
   [Releasing](#releasing)) moves `stable` automatically; a brand-new tag may
   need activating once under RTD → *Versions*.
2. **Mark version-gated features.** Documenting something that isn't in the
   current `stable` release? Add a durable "New in X.Y" admonition (see the
   remote-transcription pages for the pattern), worded as a lasting fact —
   "available from 2.0 onward", not "not yet released" — so it stays correct
   after the release ships.

ReadTheDocs' non-stable notification (RTD → *Addons → Notifications*) also warns
readers when they're viewing `latest` rather than a release.

## Releasing

Releases are automated and driven entirely by **git tags** (see
[versioning](#versioning)):

1. Update [`CHANGELOG.md`](changelog.md) with the new version's notes.
2. Tag the release and push the tag:

    ```bash
    git tag -a v0.1.0 -m "syncalong 0.1.0"
    git push origin v0.1.0
    ```

3. The `release` workflow builds the sdist + wheel, publishes them to
   [PyPI](https://pypi.org/project/syncalong/) via Trusted Publishing (OIDC, no
   stored token), and creates a GitHub Release with the artifacts attached.

Tags should be of the form `vMAJOR.MINOR.PATCH`.

## Versioning

The version is **not** stored in the source tree. It is derived from the latest
git tag by [setuptools-scm](https://setuptools-scm.readthedocs.io/):

- a checkout at tag `v0.1.0` builds as version `0.1.0`;
- commits after a tag build as `0.1.1.devN+g<hash>`.

So the only thing you bump is the tag.
