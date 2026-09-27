# Verification

- Six unit tests passed on CPU: 26-channel alignment and reconstruction, nominal masking/equal parameter count, gradients through displacement inputs, zero residual mean recovery, checkpoint restoration of both modes, and permutation of nonreference observations.
- Paired quick training completed for both modes (3 epochs each, 8/4/4 environments, 2 subcarriers). Best checkpoints were reloaded and comparison.json was saved.
- Geometry diagnostic completed at b=0 and b=0.1 on two environments, including shuffled geometry and zero displacement. At b=0 all three predictions agreed as expected.
- Runtime: Python 3.12, PyTorch 2.14.0+cpu. Full GPU training has not been run; no accuracy advantage is claimed.

No trained weights or test-run artifacts are included in the project archive.
