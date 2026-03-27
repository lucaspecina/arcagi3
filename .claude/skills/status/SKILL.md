---
name: status
description: "Show project status overview. Use when the user says /status or asks about project state."
disable-model-invocation: true
---

# Project Status

Show a quick overview of the project state.

## Steps

1. Read `CURRENT_STATE.md` for current capabilities
2. Read `TODO.md` for task status
3. Check `git log --oneline -5` for recent activity
4. Run `pytest --co -q 2>/dev/null` to count tests (if any exist)
5. Present a concise summary in Spanish:
   - What works today
   - What's in progress
   - What's next
   - Any blockers or known issues
