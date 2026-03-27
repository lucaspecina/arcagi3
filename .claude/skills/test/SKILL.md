---
name: test
description: "Run project tests. Use when the user says /test or when validating changes before commit."
disable-model-invocation: true
---

# Run Tests

Run the project test suite.

## Steps

1. Run `pytest -x -v $ARGUMENTS` from the project root
2. If tests fail, analyze the output and suggest fixes
3. If `$ARGUMENTS` is empty, run all tests
4. Report: total tests, passed, failed, and any errors

## Examples
- `/test` — run all tests
- `/test tests/test_loader.py` — run specific test file
- `/test -k "test_evaluate"` — run tests matching pattern
