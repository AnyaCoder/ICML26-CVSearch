# CLAUDE.md

This file is the repository-level coding guide. Follow these rules before writing new code.

## Architecture Vocabulary

Use these terms consistently:

- **Module**: anything with an interface and an implementation.
- **Interface**: everything a caller must know to use a module correctly: types, invariants, ordering, configuration, and error modes.
- **Implementation**: the code inside a module.
- **Depth**: leverage at the interface. A deep module hides meaningful behaviour behind a small interface.
- **Seam**: the place where a module's interface lives; behaviour can vary there without editing callers.
- **Adapter**: a concrete implementation that satisfies an interface at a seam.
- **Leverage**: what callers gain from a deep module.
- **Locality**: what maintainers gain when knowledge, bugs, and change are concentrated in one place.

Avoid using "component", "service", "API", or "boundary" when these architecture terms are what you mean.

## Architecture Rules

- Apply the deletion test before adding or keeping a module: if deleting it removes complexity, it was pass-through code; if deleting it pushes complexity into multiple callers, it was earning its keep.
- The interface is the test surface. Do not make tests depend on internal helper details unless those helpers are the module being tested.
- One adapter is only a hypothetical seam. Two adapters make a real seam. Do not invent seams for imaginary future variation.
- Prefer deep modules with small interfaces over many shallow helpers.
- Keep proposal generation, evidence-memory logic, debug artifact recording, and experiment runners in separate modules with clear interfaces.
- Delete unused logic promptly. Do not keep compatibility shells, rough one-off scripts, or stale paths unless the user explicitly asks to preserve them.

## Python Project Structure Rules

- Group code by cohesive purpose. One file should own one concept or a tightly related set of functions/classes.
- Prefer flat package structures. Add subpackages only for real sub-domains.
- Use clear `snake_case` module names. Avoid vague names like `utils2`, `debug_tmp`, `new_test`, or broad kitchen-sink files.
- Define explicit public interfaces with `__all__` for modules intended to be imported.
- Keep heavy optional runtime dependencies behind explicit submodule imports. Importing lightweight interfaces should not load model frameworks such as Torch or Transformers.
- Use package imports for package code. Keep script-style `sys.path` handling confined to experiment entry points.
- Do not duplicate existing helpers. Search first, reuse existing modules, and deepen the right module when behaviour is spreading across callers.
- Keep generated artifacts, datasets, logs, caches, and local environments out of git.

## Editing Discipline

- Scope edits to the current branch work on top of `main`.
- Preserve mainline CVSearch behaviour unless the task explicitly changes it.
- Prefer moving/renaming rough new files into coherent packages over leaving top-level experimental scripts.
- After structural edits, run at least `python -m py_compile` on touched Python packages.
- Document runnable module entry points with `python -m package.module`, not direct paths to rough scripts.

## Debug Visualization Rules

- Feature superpixel boundaries (from 72×72 SAM3 label maps) must be upscaled with bilinear interpolation (`cv2.INTER_LINEAR`) per-mask, then thresholded, to produce smooth organic contours. Never use NEAREST resize for boundary drawing — it creates rectangular artifacts.
- `_draw_contour_boundaries` owns all boundary rendering. It accepts a label_map at any resolution, upscales each binary mask with bilinear interpolation to the target image size, and draws anti-aliased contours with `cv2.findContours` + `cv2.LINE_AA`.
- Display SLIC (run at full RGB resolution) naturally produces organic boundaries — no special upscaling needed there.
- Color fill for cluster maps still uses NEAREST-resized labels (for pixel indexing), but contour overlay always uses raw small-resolution labels fed through bilinear upscale.
- No fallback code in visualization. If data is missing or invalid, skip rendering entirely rather than drawing a degraded version.

## Evidence Memory Design Principles

- The innovation path is: AttentionGuidedWindowBuilder → VLM Verify → AttentionBoxGrounder → Layout. No fallback branches on this path.
- If attention cannot produce a valid box, skip the proposal entirely. Do not fallback to a fixed window.
- If grounding fails (attention_box missing), drop the window. Do not retain "fallback" evidence items.
- Ablation-only adapters (AcceptAllVerifier, NoOpGrounder, FixedWindowBuilder) live in a separate baselines module, not alongside the primary implementations.
- One Keeper flow: verify → ground. Do not maintain parallel "grounding-first" paths in the main module.
- Score formula is additive with no fallback bias: `vlm_score + proposal_weight * proposal_score + grounding_weight * grounding_score`.
