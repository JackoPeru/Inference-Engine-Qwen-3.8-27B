# HERMES-27 — Qwen3.8-27B inference engine

Experimental, independent C++20 / CUDA inference engine targeting the Qwen3.8-27B Uncensored NVFP4-MTP GGUF model on two NVIDIA RTX 5060 Ti 16 GB GPUs (SM120).

This repository contains a snapshot of the **code under active development**, not a production-ready serving runtime. The source is preserved in its original layout to avoid interfering with ongoing work.

## Layout

- [`hermes-27/`](hermes-27/) — primary CMake project, GGUF loader, reference model, CUDA kernels, tests, technical reports and scripts.
- [`BRIEF_CODING_AGENT_QWEN38_27B_BLACKWELL.md`](BRIEF_CODING_AGENT_QWEN38_27B_BLACKWELL.md) — project goals and milestones.
- Root-level `.py`, `.cpp`, `.cu`, `.hpp` and `.orig`/`.fixed` files — ongoing investigations, standalone experiments and patch candidates. Some duplicate files in `hermes-27/` deliberately; they are retained for traceability.

## Current validation status

Review the reports under [`hermes-27/reports/`](hermes-27/reports/) for measured hardware behavior, numerical checks and remaining limitations. A full two-GPU end-to-end runtime and multiuser serving still require integration and verification.

## Dependencies

Ubuntu 24.04, CMake, a C++20 compiler, NVIDIA CUDA toolkit with SM120 support, and NVIDIA CUTLASS for the GPU GEMM kernels. The model weights and third-party libraries are **not included**. See the project brief and the CMake configuration for details.

## Data handling

Weights, model checkpoints, private settings, caches, generated build files, and benchmark dumps are deliberately excluded from this repository.
