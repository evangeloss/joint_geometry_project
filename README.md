# Joint channel–geometry encoder

Separate experimental project. The observation simulator, normalization, mean-reference residual targets, decoder and mean fusion are retained from the relative-antenna baseline. The separate geometry encoder and FiLM modulation are replaced by direct input concatenation.

For each observation the shared CNN receives 26 channels: 8 channel values (reference and difference for two complex subcarriers) plus 18 geometry values [G0, Gr-G0, Gm-Gr]. Geometry values use wavelength units and the same antenna-pair indexing. After shared processing, observation features are averaged and decoded into an antenna-domain residual added to the mean observation.

`--geometry-mode full` uses all geometry. `--geometry-mode nominal` zeros the 12 displacement descriptors but keeps G0 and the same 26-channel architecture. Both modes have identical parameter counts. Mode is stored in model configuration and restored automatically; no hooks or manual source edits are required. No gamma/beta coefficients exist in this version.

## Kaggle

Put the contents of this folder into a NEW repository, with main/, model/, etc. at its root. Clone it, enter its root, then run:

```python
!python -m unittest discover -s tests -p "test_*.py" -v
!python -m main.compare --quick --device cuda --batch-size 4
!python -m main.compare --device cuda --batch-size 4 --seed 42
```

The last command trains both modes sequentially, 30 epochs each, with identical starting weights (same seed and parameter layout), training samples and batch order. Results are under artifacts/paired_joint_TIMESTAMP/full and nominal. comparison.json summarizes both. These are fixed-data runs to isolate architecture; fresh-epoch augmentation is not included in this first comparison.

Defaults: 200 training environments, 50 validation, 50 extrapolation, 8 subcarriers (7 adjacent pairs per environment), 8 observations, batch size 4 in compare, 20 dB SNR. Training and validation b/lambda are 0.05–0.30; extrapolation is 0.35–0.50. Validation selects checkpoints and is not an untouched test set. Use the independently seeded diagnostic below for additional evaluation. Results should not be directly compared to the two-subcarrier controlled-augmentation experiment.

Individual run:

```python
!python -m main.train --geometry-mode full --device cuda --batch-size 4
```

Diagnostic on either saved run:

```python
!python -m tests.diagnose_geometry --run-dir artifacts/paired_joint_TIMESTAMP/full --environments 50 --seed 90401 --device cuda
```

Reports normal predictions, cross-environment shuffled geometry, zeroed displacement descriptors, and the mean baseline across deformation amplitudes. Zero-displacement evaluation is an intervention on a trained model; the separately trained nominal mode is the fair model comparison. Nominal-mode predictions should be unchanged by shuffled geometry. The diagnostic writes CSV and JSON.

## Limits and checkpoints

Old FiLM checkpoints are incompatible; retrain from scratch. Dataset shape remains unchanged. The nominal rigid geometry defaults match the supplied simulator; custom hardware geometry must be passed in the same coordinate frame and units. Reference observation is state 0. Arbitrary reordering of reference state is not an invariance. The CNN still operates on flattened BS/UE antenna indices; it is not a full physical array-topology model.

This is a candidate architecture, with no established accuracy gain. Compare full versus nominal, inspect shuffled geometry, and retain the prior FiLM project as a baseline with matching data and training settings.
