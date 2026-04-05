# LULESH — Research Documentation

## What is LULESH

LULESH (Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics) is a proxy app that represents a typical hydrodynamics simulation. It solves a Sedov blast problem on an unstructured hexahedral mesh using the finite element method. Each timestep computes forces on mesh nodes, updates velocities and positions, then updates element kinematic and material state. The simulation runs for approximately 1000 timesteps. It is widely used in HPC benchmarking because it is simple but representative of real workloads.

## Research Context

This repo is a modified version of LULESH used for MSc thesis research: **"Analysis of Surrogate Models at Multiple Levels for Neural Acceleration of HPC applications"** (Bibek Panthi, UAH, advisor: Dr. Joshua Booth).

The research replaces computationally expensive functions inside LULESH with small neural network surrogate models, then measures the resulting speedup and accuracy loss. Three functions at different levels of the call stack were targeted:

| Nickname | Function | Runtime share | Inputs | Outputs |
|----------|----------|---------------|--------|---------|
| **EVD** | `CalcElemVolumeDerivative` | ~9% (1 thread) | 24 floats (x,y,z coords of 8 nodes) | 24 floats (volume derivatives dvdx/y/z) |
| **HGF** | `CalcFBHourglassForceForElems` body | ~13% (1 thread) | 74 floats | 24 floats (hourglass forces) |
| **VFE** | `CalcVolumeForceForElems` | ~59% (1 thread) | 439 floats | 3 floats (node force fx/fy/fz) |

Data collection and model inference use the **HPAC-ML** framework (`#pragma approx ml`), which is a Clang compiler extension. The compiler must be built with HPAC-ML support for this codebase to work.

## Documentation Files

- [`research_changes.md`](research_changes.md) — Detailed description of all code changes made for the research (since commit `cabb12`)
- [`workflow.md`](workflow.md) — How to build, collect data, run inference, and evaluate results
