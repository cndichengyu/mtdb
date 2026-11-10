# Contributing

Thanks for looking at MTDB. This is a research prototype, so the most useful
contributions are reproducible experiments and clearly scoped extensions.

## Before opening a PR

1. Read `ARCHITECTURE.md` (or `ARCHITECTURE_EN.md`) to see the module boundaries.
2. Run the test suite:
   ```bash
   python -m pytest tests -q
   ```
3. If you change training behaviour, include the command line and the resulting
   `results.json` diff in the PR description.
4. Keep the base trainable on CPU with the `debug` preset. New features that
   require a GPU should be optional.

## Ideas that fit this repository

- event-driven/spiking kernels for the sparse recurrent core;
- closing the loop from phase clusters back to routing/gain;
- trajectory-based STDP replay (currently consolidation is readout-heavy);
- better long-horizon memory benchmarks and binding tasks;
- real sensor front-ends (DVS, audio, touch) exporting `.npz` streams;
- larger-scale runs with power/latency measurements.

## What does not fit

Claims that this substrate alone is AGI, Transformer replacement benchmarks
without a fair protocol, or changes that break the "no backpropagation in the
base" design without a discussion first.

## Style

- Python 3.10+, type hints where useful.
- Keep modules under `mtdb/` importable without optional dependencies.
- `ruff check mtdb scripts tests --select F401,F841,F821,E9` should pass.
