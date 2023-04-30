# position-effect-correction

## Setup

1. [Install Python](https://www.python.org/downloads/)
2. [Install Poetry](https://python-poetry.org/docs/#installation)
3. [Install Poetry Environment](https://python-poetry.org/docs/basic-usage/#installing-dependencies): `poetry install --no-root`
4. Activate Poetry Environment: `poetry shell`
5. Install Pycytominer: `poetry run pip install `*<path_to_pycytominer>*, where *<path_to_pycytominer>* is a [GitHub link](https://github.com/cytomining/pycytominer/tree/c90438fd7c11ad8b1689c21db16dab1a5280de6c#installation) or a locally cloned repository.

For Linux, see

- <https://github.com/python-poetry/poetry/issues/1917#issuecomment-1380429197> if installing six fails
- <https://stackoverflow.com/a/75435100> if you get "does not contain any element" warning when running `poetry install`
