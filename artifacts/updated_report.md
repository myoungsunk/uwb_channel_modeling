# Updated Report

## Commands

### `pytest -q`

```text
ERROR: test collection failed because dependency `numpy` is not installed in this environment.
ModuleNotFoundError: No module named 'numpy'
```

### `python -m scripts.run_validation --out artifacts/updated_report.md`

```text
ModuleNotFoundError: No module named 'scripts'
```

### `python scenarios/runner.py`

```text
ModuleNotFoundError: No module named 'numpy'
```
