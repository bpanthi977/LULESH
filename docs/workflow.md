# Research Workflow

## Prerequisites

- **HPAC-ML Clang compiler** with `#pragma approx` support. The Makefile uses `clang++` with `-fapprox` and links against `-lapprox`. Without this compiler, the codebase will not build.
- **PyTorch** (for model training/inference via HPAC-ML)
- The `model_search` repo (separate repo at `/mnt/SharedOne/bpanthi/model_search`) contains training scripts, `energy.py`, and the `Optuna`-based hyperparameter search.

---

## 1. Building

Edit `CXXFLAGS` in the `Makefile` to enable the desired feature flags, then build:

```bash
make clean && make
```

### Compile-time feature flags

| Flag | Effect |
|------|--------|
| `-DEVD_COLLECT` | Enable data collection for EVD function |
| `-DEVD_COLLECT_ITERS=N` | Collect once every N timestep calls (default: every call) |
| `-DEVD_INFER` | Replace EVD function with model inference |
| `-DHGF_COLLECT` | Enable data collection for HGF function |
| `-DHGF_COLLECT_ITERS=N` | Collect once every N timestep calls |
| `-DHGF_INFER` | Replace HGF function with model inference |
| `-DE_ALL` | Enable energy array binary dump |

Do not combine `*_COLLECT` and `*_INFER` for the same function in one build. EVD and HGF flags are independent and can be combined freely.

HGF collect/infer **requires OpenMP** (`-fopenmp` already in CXXFLAGS). Run with `OMP_NUM_THREADS > 1`.

---

## 2. Data Collection

**Goal:** Run LULESH and record input/output arrays for a target function into an HDF5 file.

1. Add the collect flag to Makefile (e.g., `-DEVD_COLLECT -DEVD_COLLECT_ITERS=1`) and rebuild.
2. Set the output DB path and run:

```bash
HPAC_DB_FILE=/path/to/EVD.h5 ./lulesh2.0 -s 30
```

The `-s 30` sets the grid size (default is 30×30×30). Each timestep appends a batch of `numElem` samples to the HDF5 file.

**Throttling:** For large functions (HGF, VFE), collecting every iteration produces very large datasets. Use `COLLECT_ITERS=10` or higher to sample every Nth call.

**Multiple runs for sanity check:** Run twice with different seeds or slight variations and compare dataset statistics. Discrepancies in mean/variance of features indicate annotation bugs.

---

## 3. Model Inference

**Goal:** Run LULESH with a trained model replacing a target function.

1. Add the infer flag to Makefile (e.g., `-DEVD_INFER -DE_ALL`) and rebuild.
2. Point to the model and run:

```bash
SURROGATE_MODEL=/path/to/model.pt \
ENERGY_DUMP_FILE_NAME=Energy-mymodel-30.bin \
ENERGY_DUMP_TYPE=last \
./lulesh2.0 -s 30
```

### Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `SURROGATE_MODEL` | HPAC-ML | Default model path (used when no per-function var is set) |
| `EVD_MODEL_PATH` | HPAC-ML | Model path specifically for EVD function |
| `HGF_MODEL_PATH` | HPAC-ML | Model path specifically for HGF function |
| `HPAC_DB_FILE` | HPAC-ML | Output HDF5 file path for data collection |
| `ENERGY_DUMP_FILE_NAME` | lulesh.cc | Binary file to write energy arrays to (requires `-DE_ALL`) |
| `ENERGY_DUMP_TYPE` | lulesh.cc | `all` (every iteration) or `last` (final state only) |

When running two models simultaneously (e.g., EVD + HGF), use `EVD_MODEL_PATH` and `HGF_MODEL_PATH` to point to separate model files.

---

## 4. Generating the Reference Energy File

Before evaluating any model, generate a baseline energy file from an unmodified run:

```bash
# Build without any INFER flags, but with -DE_ALL
ENERGY_DUMP_FILE_NAME=Energy_Original-30.bin \
ENERGY_DUMP_TYPE=last \
./lulesh2.0 -s 30
```

This file is the ground truth for accuracy evaluation.

---

## 5. Evaluating Accuracy

Use the `energy_mae.sh` convenience script:

```bash
./energy_mae.sh /path/to/model.pt 30
```

This runs the model, records execution time, and computes energy MAE vs the reference. Results are saved in `<model_dir>-30/energy_mae.txt` and `execution_time.txt`.

For a custom output directory or limited iterations:

```bash
./energy_mae.sh /path/to/model.pt 30 my_experiment_name 1000
```

To compute MAE manually (using `energy.py` from the `model_search` repo):

```bash
python energy.py \
  --original Energy_Original-30.bin \
  --model Energy-mymodel-30.bin \
  --visualize
```

**Acceptance threshold:** Energy MAE < 2.0 (mean energy value is ~239).

---

## 6. LULESH Command-line Options

```
./lulesh2.0 -s <grid_size> -i <max_iters> -p
```

| Option | Description |
|--------|-------------|
| `-s N` | Grid size (N×N×N elements, default 30) |
| `-i N` | Max iterations (default: run until simulation time 0.01) |
| `-p` | Print progress every iteration |
