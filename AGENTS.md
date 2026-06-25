# AGENTS.md

This file is the repository-level coding guide for future agent work.

## Code Action Guide

When changing this repository, use these rules before writing new code.

### Architecture Vocabulary

Use these terms consistently:

- **Module**: anything with an interface and an implementation.
- **Interface**: everything a caller must know to use a module correctly: types, invariants, ordering, configuration, and error modes.
- **Implementation**: the code inside a module.
- **Depth**: leverage at the interface. A deep module hides meaningful behaviour behind a small interface. A shallow module exposes nearly as much complexity as it contains.
- **Seam**: the place where a module's interface lives; behaviour can vary there without editing callers.
- **Adapter**: a concrete implementation that satisfies an interface at a seam.
- **Leverage**: what callers gain from a deep module.
- **Locality**: what maintainers gain when knowledge, bugs, and change are concentrated in one place.

Avoid using "component", "service", "API", or "boundary" when these architecture terms are what you mean.

### Architecture Rules

- Apply the deletion test before adding or keeping a module: if deleting it removes complexity, it was pass-through code; if deleting it pushes complexity into multiple callers, it was earning its keep.
- The interface is the test surface. Do not make tests depend on internal helper details unless those helpers are the module being tested.
- One adapter is only a hypothetical seam. Two adapters make a real seam. Do not invent seams for imaginary future variation.
- Prefer deep modules with small interfaces over many shallow helpers.
- Keep proposal generation, evidence-memory logic, debug artifact recording, and experiment runners in separate modules with clear interfaces.
- Delete unused logic promptly. Do not keep compatibility shells, rough one-off scripts, or stale paths unless the user explicitly asks to preserve them.
- When asked for an architecture review, inspect the relevant domain docs and ADRs first, then report architecture candidates in terms of Module, Interface, Depth, Seam, Adapter, Leverage, and Locality.

### Python Project Structure Rules

- Group code by cohesive purpose. One file should own one concept or a tightly related set of functions/classes.
- Prefer flat package structures. Add subpackages only for real sub-domains.
- Use clear `snake_case` module names. Avoid vague names like `utils2`, `debug_tmp`, `new_test`, or broad kitchen-sink files.
- Define explicit public interfaces with `__all__` for modules intended to be imported.
- Keep heavy optional runtime dependencies behind explicit submodule imports. Importing lightweight interfaces should not load model frameworks such as Torch or Transformers.
- Use package imports for package code. Keep script-style `sys.path` handling confined to experiment entry points that must interoperate with the existing CVSearch script layout.
- Do not duplicate existing helpers. Search first, reuse existing modules, and deepen the right module when behaviour is spreading across callers.
- Keep generated artifacts, datasets, logs, caches, and local environments out of git.

### Editing Discipline

- Scope edits to the current branch work on top of `main`.
- Preserve mainline CVSearch behaviour unless the task explicitly changes it.
- Prefer moving/renaming rough new files into coherent packages over leaving top-level experimental scripts.
- After structural edits, run at least `python -m py_compile` on touched Python packages. Do not manually clean `__pycache__`; Python will overwrite it on the next run.
- Document runnable module entry points with `python -m package.module`, not direct paths to rough scripts.
