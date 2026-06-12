# Codex refactoring rules

Goal: make this repository smaller, clearer, and easier to maintain without changing public behavior.

Hard constraints:
- Do not change public APIs unless explicitly requested.
- Do not change numerical results except within documented tolerances.
- Do not rewrite large subsystems in one PR.
- Prefer deleting dead code, merging duplicated logic, simplifying files, and removing unused abstractions.
- Keep every PR small and reviewable.
- Every change must pass tests, linting, and type checks.
- If behavior changes are necessary, stop and explain before editing.

Definition of done:
- Tests pass.
- New or updated tests cover the refactored behavior.
- Public imports still work.
- Numerical regression tests pass.
- LOC, complexity, or duplication improves.
- PR description explains what was removed, what was preserved, and how correctness was checked.

For JAX code:
- Do not accidentally move dynamic values into static arguments.
- Do not introduce recompilation loops.
- Do not change dtype behavior without tests.
- Do not change array shapes, pytree structure, or JIT boundaries unless explicitly asked.
- Do not optimize for elegance at the cost of compile time or memory use.
