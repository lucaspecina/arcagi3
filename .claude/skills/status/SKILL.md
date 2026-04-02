---
name: status
description: "Show project status overview. Use when the user says /status or asks about project state."
disable-model-invocation: true
---

# Project Status

Show a quick overview of the project state.

## Steps

1. Read `CURRENT_STATE.md` for current capabilities
2. Read `TODO.md` for operational board (NOW/NEXT/BLOCKED)
3. Check active issues in `issues/` (status: active)
4. Check `git log --oneline -5` for recent activity
5. Run `pytest --co -q 2>/dev/null` to count tests (if any exist)
6. Check `AUTORESEARCH.md` status (ON/OFF)
7. Present a concise summary in Spanish:
   - What works today
   - Active issues and their status
   - What's in NOW and NEXT
   - Any blockers or known issues
   - Autoresearch status
