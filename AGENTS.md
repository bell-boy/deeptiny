# AGENTS Notes

## Workflow reminders
- Commit after finishing changes in a task.
- If needed, rebase off `main` before committing.

## Repo/worktree notes
- This workspace is a git worktree from a larger repo.
- Keep `main` as stable source of truth and coordinate feature branches/worktrees accordingly.

## Project conventions and preferences
- Prioritize performance in math kernels; use `cblas` fast paths where they fit safely.
- Keep math layering clear: `math.cc` handles broadcast + device routing + autograd wiring, while device kernels do raw `TensorImpl` math.
- In-place math ops are forbidden on destination tensors with any zero stride (broadcasted views).
- For backward functions that save tensors, use `Context` and `ContextObjects` enum keys:
  - `enum struct ContextObjects : uint64_t { ... }`
- For elementary math changes, maintain forward/backward/in-place test coverage and guard tests for invalid mutation/version behavior.
