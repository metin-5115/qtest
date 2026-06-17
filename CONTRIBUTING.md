# Contributing to qtest

First off — thank you for taking the time to contribute! This document
walks you through everything you need to know to get a development
environment up, make a change, run the test suite, and open a pull
request.

By participating in this project you agree to abide by our
[Code of Conduct](https://github.com/metin-5115/qtest/blob/main/CODE_OF_CONDUCT.md).

---

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Development environment](#development-environment)
- [Project layout](#project-layout)
- [Running the test suite](#running-the-test-suite)
- [Code style](#code-style)
- [Building the documentation](#building-the-documentation)
- [Submitting a pull request](#submitting-a-pull-request)
- [Reporting bugs](#reporting-bugs)
- [Proposing features](#proposing-features)
- [Security issues](#security-issues)

---

## Ways to contribute

There are many ways to help — and writing code is only one of them:

- **Reporting bugs** and producing minimal reproducers.
- **Improving documentation** — typos, clearer examples, missing
  cross-references.
- **Triaging issues** — reproducing reported bugs, asking the
  clarifying questions.
- **Writing tests** — every additional property test makes the library
  more trustworthy.
- **Implementing features** — see the issue tracker for ones tagged
  `good first issue` and `help wanted`.

If in doubt, [open an issue](https://github.com/metin-5115/qtest/issues)
to talk through your idea before writing code.

---

## Development environment

We support Python 3.9 – 3.12 on Linux, macOS, and Windows. You'll need
a working Python install and `git`.

```bash
git clone https://github.com/metin-5115/qtest.git
cd qtest

python -m venv .venv
source .venv/bin/activate           # Linux / macOS
.venv\Scripts\activate              # Windows PowerShell

pip install -e ".[dev]"
pre-commit install
```

The `dev` extra installs the formatter (`black`), linter (`ruff`),
type checker (`mypy`), test runner (`pytest`, `pytest-cov`,
`pytest-xdist`), and release tooling. `pre-commit install` wires up
the formatting / linting hooks so each commit is pre-checked.

---

## Project layout

```
qtest/
├── qtest/              # Library source code
│   ├── assertions/     # The four core assertions
│   ├── backends/       # Backend abstraction (Qiskit, Cirq, PennyLane)
│   ├── fixtures/       # pytest fixtures and plain factories
│   ├── metrics/        # Distances and statistical tests
│   ├── strategies/     # Hypothesis strategies
│   ├── plugin.py       # pytest plugin entry point
│   └── config.py
├── tests/              # Test suite (pytest)
├── examples/           # Runnable, documented usage examples
├── docs/               # Sphinx documentation
└── pyproject.toml      # Build, deps, and tool configuration
```

---

## Running the test suite

A full local run is one command:

```bash
pytest
```

Useful variations:

```bash
pytest -m "not slow"                  # skip slow tests
pytest tests/test_assertions.py -v    # one file, verbose
pytest -n auto                        # parallel (via pytest-xdist)
pytest --cov=qtest --cov-report=term  # coverage report
```

Coverage is enforced on CI at **≥ 85 %** for the project as a whole.
PRs that drop coverage below that line will fail CI.

You can run the full quality suite (lint + types + tests) with
[`tox`](https://tox.wiki/):

```bash
tox            # all environments declared in tox.ini
tox -e py311   # one Python version
tox -e lint    # ruff + black --check
tox -e type    # mypy
```

---

## Code style

We let tooling do the bikeshedding so reviewers can focus on the code:

| Tool       | Purpose                          | Config                          |
| ---------- | -------------------------------- | ------------------------------- |
| **black**  | Code formatter (line length 100) | `pyproject.toml → [tool.black]` |
| **ruff**   | Linter + import sorter           | `pyproject.toml → [tool.ruff]`  |
| **mypy**   | Static type checker (strict)     | `pyproject.toml → [tool.mypy]`  |

Run them all locally before pushing:

```bash
black .
ruff check . --fix
mypy
```

A few conventions on top of the tooling:

- **Type-hint everything.** `mypy` is in strict mode; new code without
  hints will fail CI.
- **Docstrings are NumPy-style** (parsed by `sphinx.ext.napoleon`).
  Every public function, class, and module gets one.
- **Tests live in `tests/`**, mirror the package layout, and use
  `pytest`'s function-style API. No `unittest.TestCase` classes.
- **No commented-out code in commits.** Delete it; `git` remembers.

---

## Building the documentation

Install the docs toolchain and run an autobuild server:

```bash
pip install -e ".[docs]"
sphinx-autobuild docs docs/_build/html
```

The site will be served at <http://127.0.0.1:8000> and live-reload on
every save. For a single one-shot build:

```bash
cd docs
make html         # Linux / macOS
.\make.bat html   # Windows
```

---

## Submitting a pull request

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feat/short-descriptive-name
   ```
2. **Make your changes** in small, focused commits.
3. **Add or update tests** that exercise the change. PRs without tests
   for new behaviour are unlikely to be merged.
4. **Update `CHANGELOG.md`** under the `[Unreleased]` section.
5. **Update documentation** if you changed user-facing behaviour.
6. **Run the full local check** before pushing:
   ```bash
   black . && ruff check . && mypy && pytest
   ```
7. **Open the pull request** against `main`. Fill in the PR template —
   what changes, why, and how you tested it.
8. **Respond to review feedback.** Pushes to your branch update the PR
   automatically; we'll squash on merge.

### Commit message style

We use [Conventional Commits](https://www.conventionalcommits.org/)
loosely. Examples:

```
feat(strategies): add Haar-random unitary strategy
fix(assertions): correct phase handling in assert_state_close
docs: clarify --qtest-seed semantics
chore(ci): bump actions/checkout to v4
```

This isn't strictly enforced, but it makes the changelog easier to
write at release time.

---

## Reporting bugs

A great bug report has:

1. A short, descriptive title.
2. The qtest version (`pip show qtest`), Python version, and OS.
3. A **minimal reproducer** — the smallest possible test that triggers
   the bug. If the bug surfaces via a Hypothesis property, paste the
   shrunk blob Hypothesis prints.
4. What you expected to happen, and what actually happened.
5. The full traceback (not a screenshot).

Open the issue at <https://github.com/metin-5115/qtest/issues>.

---

## Proposing features

Open an issue *before* writing significant code. We'd rather discuss
design up front than ask you to rework a finished PR. A good feature
proposal includes:

- The **problem** you're trying to solve, ideally with a concrete
  user-side example.
- The **API** you have in mind (a snippet of the calling code is worth
  pages of prose).
- **Alternatives** you considered and why you rejected them.

---

## Security issues

**Do not file security issues in the public issue tracker.** See
[SECURITY.md](https://github.com/metin-5115/qtest/blob/main/SECURITY.md) for the disclosure process.

---

Thanks again for contributing to qtest!
