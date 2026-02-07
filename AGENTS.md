# AGENTS Notes

## Workflow reminders
- Commit after finishing changes in a task, then open/update a PR for user review.
- If needed, rebase off `main` before committing.
- After each commit, update `AGENTS.md` if there are durable decisions/conventions worth preserving.

## Repo/worktree notes
- This workspace is a git worktree from a larger repo.
- Keep `main` as stable source of truth and coordinate feature branches/worktrees accordingly.

## Project conventions and preferences
- Prioritize performance in math kernels; use `cblas` fast paths where they fit safely.
- Keep math layering clear: `math.cc` handles broadcast + device routing + autograd wiring, while device kernels do raw `TensorImpl` math.
- Keep compute dispatch centralized in `src/dispatch/` (kernel routing + non-autograd tensor outputs) and keep graph wiring in `math.cc`/`functional.cc`/backward `Function` classes.
- In math/autograd wiring, use `dispatch::binary::Op` directly instead of mirroring duplicate local binary-op enums/mapping helpers.
- Keep dispatch headers split by op (`src/dispatch/<op>.h`) with `src/dispatch/dispatch.h` as a small umbrella include header.
- Organize both dispatch and CPU kernels by operation where practical (`src/dispatch/<op>.cc`, `src/cpu/kernels/<op>.cc`), with shared helpers only when they reduce duplication cleanly.
- Keep `Tensor`/`TensorImpl` boundary ergonomic: allow implicit `TensorImpl -> Tensor` (null autograd) and `Tensor -> TensorImpl` conversions to avoid verbose accessor boilerplate in dispatch-heavy code.
- In-place math ops are forbidden on destination tensors with any zero stride (broadcasted views).
- For backward functions that save tensors, use `Context` and `ContextObjects` enum keys:
  - `enum struct ContextObjects : uint64_t { ... }`
- Keep storage-version lookup encapsulated on `Context` (member helper), not as a free-floating helper.
- Keep kernel read paths version-safe: when inputs are read-only, access storage through const pointers so read access does not bump storage version counters used by `Context` checks.
- Higher-order gradients are intentionally unsupported: `Tensor::Backward()` does not accept `keep_graph`, `Engine::Run` executes under no-grad, and backward implementations should use dispatch/kernel paths rather than graph-building frontend ops.
- Treat Function parent presence as an invariant: assert parents exist in backward/graph traversal paths instead of silently skipping nulls.
- Reuse shared helpers in `tests/test_utils` (e.g., tensor construction/data assertions) instead of adding ad-hoc per-test utilities when the helpers are broadly reusable.
- For elementary math changes, maintain forward/backward/in-place test coverage and guard tests for invalid mutation/version behavior.
- Keep `math::BatchedMatMul` semantics standard: broadcast leading dims only, and require exact contracted dim match after applying transpose flags.
- Keep `Tensor::Reshape` differentiable but view-only: require contiguous input and matching element count.
- Keep `Tensor::Squeeze` API minimal: expose only the `std::vector<uint64_t>` overload and rely on implicit brace-init conversion at call sites.
- Keep `functional::Reduce` API minimal: expose only the `std::vector<uint64_t>` overload and rely on implicit brace-init conversion at call sites.
- Keep tensor creation APIs on `Tensor` (`CreateUniform`, `Zeros`, `FromVector`) and keep `functional` focused on transform/reduction ops.
- Keep `Tensor::CreateUniform` and `Tensor::Zeros` trainability explicit via a `requires_grad` parameter (default `false`).
- Keep autograd pending-count bookkeeping owned by `Engine`; avoid exposing mutators on `AutogradMeta` for `pending_`.
- Keep autograd interfaces minimal: `updateGrad` only accumulates gradients, and backward `Function` callbacks should not take `Engine` unless it is truly needed.
- Keep slice semantics explicit: `Tensor::operator()` returns `TensorSliceProxy` for mutable slicing and a read `Tensor` via conversion, and slice assignment autograd must be owned by the destination tensor metadata (not a temporary slice object).
- Keep CMake presets split by intent: `dev` for debug/testing, and `release` for optimized builds with `DEEPTINY_BUILD_TESTS=OFF` (output in `build/release`).
- Keep local + CI build/test execution centralized via `scripts/ci-local.sh`; Docker workflows should call that script instead of duplicating command sequences.
- Keep deeptiny as a pure FetchContent integration target; do not add install/export package metadata for `find_package`.
- Embedded use should default `DEEPTINY_BUILD_TESTS=OFF` and `DEEPTINY_ENABLE_WERROR=OFF` unless the parent explicitly enables them.
- Keep standalone demos under `demo/<name>` as independent CMake projects that consume Deep Tiny via `FetchContent` with a pinned commit (`DEEPTINY_GIT_TAG`), and allow local override through `FETCHCONTENT_SOURCE_DIR_DEEPTINY` when needed.
- Keep `transfomer-demo` embedding contract strict: `Embedding::operator()(indices, shape)` requires `indices.size() == product(shape)`, throws on out-of-range indices, and returns `shape + {embedding_dim}`.
