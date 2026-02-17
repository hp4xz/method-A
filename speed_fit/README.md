# speed_fit

This folder contains alternative drivers focused on throughput:

- `fast_method_a_fit.py`: `FastMethodAFit` with an iterative (stack-based) `FillChannels` implementation to avoid Python recursion overhead.
- `fast_fit_driver.py`: fitting and batch APIs:
  - `fit_noE_fast(...)`
  - `fit_noE_batch(..., workers=N)` for serial or multiprocess batch processing.
- `speed-fit.ipynb`: notebook entry point for quick sanity and batch examples.

## Notes

- Physics equations and parameter conventions match the existing Method A workflow.
- Quick sanity mode is optimized for speed (`quick_sanity=True`) and performs one simulation without minimization.
