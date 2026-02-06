# AGENTS Notes

## Workflow reminders
- Commit after finishing changes in a task.
- If needed, rebase off `main` before committing.
- After each commit, update `AGENTS.md` if there are durable decisions/conventions worth preserving.

## Repo/worktree notes
- This workspace is a git worktree from a larger repo.
- Keep `main` as stable source of truth and coordinate feature branches/worktrees accordingly.

## Project conventions and preferences
- Prioritize performance in math kernels; use `cblas` fast paths where they fit safely.
- Keep math layering clear: `math.cc` handles broadcast + device routing + autograd wiring, while device kernels do raw `TensorImpl` math.
- In-place math ops are forbidden on destination tensors with any zero stride (broadcasted views).
- For backward functions that save tensors, use `Context` and `ContextObjects` enum keys:
  - `enum struct ContextObjects : uint64_t { ... }`
- Keep storage-version lookup encapsulated on `Context` (member helper), not as a free-floating helper.
- Treat Function parent presence as an invariant: assert parents exist in backward/graph traversal paths instead of silently skipping nulls.
- Reuse shared helpers in `tests/test_utils` (e.g., tensor construction/data assertions) instead of adding ad-hoc per-test utilities when the helpers are broadly reusable.
- For elementary math changes, maintain forward/backward/in-place test coverage and guard tests for invalid mutation/version behavior.
- Keep `Tensor::Squeeze` API minimal: expose only the `std::vector<uint64_t>` overload and rely on implicit brace-init conversion at call sites.
- Keep autograd pending-count bookkeeping owned by `Engine`; avoid exposing mutators on `AutogradMeta` for `pending_`.
- Keep autograd interfaces minimal: `updateGrad` only accumulates gradients, and backward `Function` callbacks should not take `Engine` unless it is truly needed.
- Keep CMake presets split by intent: `dev` for debug/testing, and `release` for optimized builds with `BUILD_TESTING=OFF` (output in `build/release`).
- Keep CMake package/export support working: install must provide `deeptinyConfig.cmake` and exported target `deeptiny::deeptiny` for `find_package(deeptiny CONFIG REQUIRED)`.
