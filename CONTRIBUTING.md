# Contribution guidelines

## Reporting issues

Report bugs and feature requests through the
[issue tracker](https://github.com/cledouarec/doc-checker/issues).
Before opening a new issue, check that it has not already been reported.

## Pull requests

- Create a dedicated branch from `main`.
- Keep changes focused: one topic per pull request.
- Make sure `uv run pytest` and `uv run pre-commit run --all-files` pass
  before submitting.

## Commit Message Format

This project follows the
[Conventional Commits](https://www.conventionalcommits.org/) specification,
enforced by [Commitizen](https://commitizen-tools.github.io/commitizen/).

### Commit Message Header

```text
<type>(<scope>): <short summary>
```

#### Type

| Type       | Description                                        |
| ---------- | -------------------------------------------------- |
| `feat`     | A new feature                                      |
| `fix`      | A bug fix                                          |
| `docs`     | Documentation only changes                         |
| `style`    | Formatting changes without code meaning            |
| `refactor` | A change that neither fixes a bug nor adds a feature |
| `perf`     | A change that improves performance                 |
| `test`     | Adding or correcting tests                         |
| `build`    | Changes to the build system or dependencies        |
| `ci`       | Changes to CI configuration                        |
| `chore`    | Other changes that don't modify src or test files  |

### Updating the changelog

The changelog is generated automatically by Commitizen from the commit
history during the release workflow — do not edit it by hand.

## Developing

### Set up

Install [mise](https://mise.jdx.dev/), then from the repository root:

```bash
mise install                # installs Python 3.14 and uv
uv sync                     # creates .venv and installs all dependencies
uv run pre-commit install   # installs the git hooks
```

Run the test suite:

```bash
uv run pytest
```
