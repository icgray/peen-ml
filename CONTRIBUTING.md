# Contributing to peen-ml

Thank you for your interest in contributing to peen-ml!

## How to report a bug

1. Search the [issue tracker](../../issues) to check whether the bug has already been reported.
2. If not, open a new issue and include:
   - A short description of the bug
   - Steps to reproduce (dataset size, OS, Python/PyTorch version)
   - Expected behaviour vs. actual behaviour
   - Any error messages or tracebacks

## How to propose a feature

Open an issue with the label `enhancement`. Describe:
- The use case motivating the feature
- What the proposed API or behaviour would look like
- Any relevant references (papers, existing tools)

## How to submit a pull request

1. Fork the repository and create a branch: `git checkout -b my-feature`
2. Install in editable mode with test dependencies: `pip install -e ".[test]"`
3. Make your changes. Run the test suite before committing:
   ```bash
   pytest tests/ -q
   ```
4. Follow PEP 8 style (checked by the CI pylint workflow).
5. Open a pull request against `main`. Describe what the PR changes and why.
6. A maintainer will review within a few days.

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Keep lines under 100 characters.
- Add docstrings to all public functions and classes.
- Use type hints where practical.

## Running the tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run full test suite
pytest tests/ -q

# Run with coverage report
pytest tests/ --cov=src/peen-ml --cov-report=term-missing
```

## Seeking support

- Open a [GitHub Discussion](../../discussions) for general questions.
- Open a [GitHub Issue](../../issues) for bugs or feature requests.
- For questions about the physics model (Shen & Atluri 2006), see `src/peen-ml/impact_sim.py` and the references in `paper.bib`.
