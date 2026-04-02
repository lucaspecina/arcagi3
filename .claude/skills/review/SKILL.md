---
name: review
description: "Review recent changes before commit. Use when the user says /review or before presenting changes for approval."
disable-model-invocation: true
---

# Code Review

Review recent changes for quality, correctness, and alignment with project goals.

## Steps

1. Run `git diff` to see all unstaged changes
2. Run `git diff --cached` to see staged changes
3. Read changed files for full context
4. Review against these criteria:
   - **Correctness:** Does the code do what it claims?
   - **LA PREGUNTA alignment:** Does this help answer the guiding question?
   - **Conventions:** Follows project code conventions (CLAUDE.md)?
   - **Tests:** Are there tests for new functionality?
   - **Security:** No hardcoded credentials, no injection vectors
5. Run `pytest -x -v` to verify tests pass
6. Run `ruff check .` to verify linting
7. Check which docs need updating (trigger table in CLAUDE.md)
8. Present findings in Spanish:
   - Summary of changes
   - Any issues found
   - Docs that need updating
   - Recommendation: ready to commit or needs fixes
