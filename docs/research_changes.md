# Research Changes (since commit cabb12)

All changes described here were added on top of the upstream LULESH 2.0 codebase. The first research commit is `0a54831` ("Wrap CalcForceForNodes with approx"); the base commit before any research work is `cabb12`.

---

## New Files: `approx.cc` / `approx.h`

These files implement a generic C++ wrapper for surrogate model data collection and inference, used for the VFE-level experiment (`CalcForceForNodes`).

**`approx.h`** defines the `ApproxConfig` struct:

```c
typedef struct {
  const char* name;        // label used in HPAC-ML
  const char* model_path;  // path to .pt model file
  const char* db_path;     // path to HDF5 output file
  const int collect_every; // collect 1 in every N calls
  bool collect;            // enable data collection
  bool infer;              // enable model inference
  int input_dim;           // # of doubles per sample (input)
  int output_dim;          // # of doubles per sample (output)
  int (*get_N)(void*);             // returns number of samples in this call
  void (*fill_input)(void*, double*);   // copy inputs into flat array
  void (*fill_output)(void*, double*);  // copy outputs into flat array
  void (*apply_output)(void*, double*); // write model output back to domain
  int funcall_counter;     // incremented each call
} ApproxConfig;
```

**`approx.cc`** implements `wrap_approx()`, which:
1. Checks if this call should be collected (based on `collect_every` throttle)
2. If collecting: calls the original function, then uses `#pragma approx ml(offline)` to store inputs/outputs to HDF5
3. If inferring: uses `#pragma approx ml(infer)` to run the model, then calls `apply_output` to write results back
4. Otherwise: calls the original function unchanged

`approx.cc` also defines the specific config `Config_CalcForceForNodes` for the VFE-level experiment, with input layout `ln_input` (per-node struct with properties of the node and its 8 neighboring elements). This VFE wrapper was used for early exploration but the VFE function proved too complex to model well — see thesis results.

**Why EVD and HGF don't use this wrapper:** EVD and HGF operate on batches of elements and their #pragma annotations are placed inline in `lulesh.cc`, which gives more control over the tensor layout needed for the HPAC-ML tensor functors.

---

## EVD Annotations (`lulesh.cc`, around line 1180)

Located inside `CalcHourglassControlForElems`, after the per-element loop that collects `x8n/y8n/z8n` and computes `dvdx/dvdy/dvdz`.

**Compile-time flags:**
- `-DEVD_COLLECT` — enable data collection mode
- `-DEVD_INFER` — enable model inference mode
- `-DEVD_COLLECT_ITERS=N` — collect once every N calls (default should be set in Makefile comment)

When `EVD_INFER` is defined, the original `CalcElemVolumeDerivative` call is skipped (guarded with `#ifndef EVD_INFER`); the model output is written into `dvdx/dvdy/dvdz` instead.

**Tensor layout:**

The input tensor maps three 2D arrays (`x8n`, `y8n`, `z8n`, each `[numElem][8]`) into a single `[numElem][24]` tensor with x/y/z grouped: `[x0..x7, y0..y7, z0..z7]` for each element. This grouping is intentional — domain knowledge shows that x, y, z coordinates are processed in independent groups first, so a wide network with 3 groups can exploit this.

```c
#pragma approx declare tensor_functor(i3_map: [i, j] = ([i, j], [i, j], [i, j]))
#pragma approx declare tensor(input: i3_map(x8n[0:numElem][0:8], y8n[0:numElem][0:8], z8n[0:numElem][0:8]))
```

Output is a flat `out[numElem][24]` array with layout `[dvdx0..7, dvdy0..7, dvdz0..7]` per element.

**Model path:** set via `EVD_MODEL_PATH` env var or `SURROGATE_MODEL` env var.

---

## HGF Annotations (`lulesh.cc`, around line 880)

Located inside `CalcFBHourglassForceForElems`, wrapping the parallelized element loop.

**Requires OpenMP** (`numthreads > 1`). An assertion fires if you try to use HGF_COLLECT/INFER in single-threaded mode because the `f_elem` array (needed as a flat output buffer) is only allocated in multi-threaded mode.

**Compile-time flags:**
- `-DHGF_COLLECT` — enable data collection mode
- `-DHGF_INFER` — enable model inference mode
- `-DHGF_COLLECT_ITERS=N` — collect once every N calls

**Tensor layout (74 inputs per element):**

```c
#pragma approx declare tensor_functor(ipmap: [i, 0:74] = (
    [i,_],  [i, _],  [i, _],   // x8n, y8n, z8n     (8+8+8 = 24: node coords)
    [i,_],  [i, _],  [i, _],   // dvdx, dvdy, dvdz   (8+8+8 = 24: volume derivs)
    [i,_],  [i, _],  [i, _],   // xd8, yd8, zd8      (8+8+8 = 24: node velocities)
    [i],                        // coeff               (1: hourglass coefficient)
    [i]))                       // determ              (1: element volume)
```

The `coeff` value is computed from `hourg * ss * elemMass / volume^(1/3)`. The `xd8/yd8/zd8` arrays are pre-gathered from domain node velocities into per-element arrays before the pragma block.

Output is `forces[N][24]` — hourglass forces for 8 nodes (hgfx, hgfy, hgfz interleaved), cast from the `f_elem` XYZ struct array.

**Model path:** set via `HGF_MODEL_PATH` env var or `SURROGATE_MODEL` env var.

---

## SFD Annotations (`lulesh.cc`, around line 540)

`CalcElemShapeFunctionDerivatives` (nickname SFD) was an early exploration target. The annotations are present (`SFD_COLLECT`, `SFD_INFER` flags) but this function was not pursued in the final research. The model path is hardcoded to an absolute path — do not use this in production.

---

## Energy Dump (`lulesh.cc` main loop, `lulesh-util.cc`)

To compare surrogate model accuracy against the unmodified run, the per-element energy array is written to a binary file at each timestep (or just the last).

**Requires compile flag:** `-DE_ALL`

**Environment variables:**
- `ENERGY_DUMP_FILE_NAME` — output file path (default: `Energy.bin`)
- `ENERGY_DUMP_TYPE` — `all` (write every iteration) or `last` (write only final state, default: `all`)

**File format** (written by `WriteArrayToFile` in `lulesh-util.cc`):
```
int32  n_dims          (= 3)
int32  dims[3]         (= nx, nx, nx)
float64[nx*nx*nx]  energy values
```
For `ENERGY_DUMP_TYPE=all`, one such block is appended per timestep.

**ModifiedMaxRelDiff:** An additional sanity metric `ModifiedMaxRelDiff` was added to `VerifyAndWriteFinalOutput`. It computes max relative diff with a small epsilon in the denominator (`AbsDiff / (e + 1e-3)`) to avoid division by near-zero energy values that inflate `MaxRelDiff`.

---

## Makefile Changes

- Compiler changed from `g++` to `clang++` (required for HPAC-ML `#pragma approx` support)
- Added `-fapprox` compile flag and `-lapprox` link flag (HPAC-ML library)
- `approx.cc` added to the source list
- Compile-time feature flags documented in comments at top of Makefile:

```makefile
# -DHGF_COLLECT -DHGF_COLLECT_ITERS=10
# -DEVD_COLLECT -DEVD_COLLECT_ITERS=10
# -DEVD_INFER
# -DHGF_INFER
```

To enable a feature, add the flag to `CXXFLAGS`. Multiple flags can be combined (e.g., both `EVD_COLLECT` and `HGF_COLLECT` at once, but not `*_COLLECT` and `*_INFER` together for the same function).

---

## `energy_mae.sh`

A convenience script to run LULESH with a model and compute energy MAE vs a reference run.

```
Usage: ./energy_mae.sh <path_to_model.pt> <grid_size> [experiment_dir] [max_iters]
```

It:
1. Runs `./lulesh2.0` with `SURROGATE_MODEL` and `ENERGY_DUMP_TYPE=last`
2. Records wall-clock execution time to `<experiment_dir>/execution_time.txt`
3. Calls `energy.py` from the `model_search` repo to compute MAE and save to `<experiment_dir>/energy_mae.txt`

The reference energy file `Energy_Original-<grid_size>.bin` must exist (generated by running LULESH without any model with `ENERGY_DUMP_TYPE=last`). The script paths (`LULESH_DIR`, `SCRIPT_DIR`) are hardcoded to the researcher's machine — update these if running elsewhere.
