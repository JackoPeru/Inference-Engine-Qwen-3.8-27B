# P1 Block0 — corrected reference (2026-11-13)

The legacy STAGE3/STAGE4 reference files /tmp/e1_gl.bin, /tmp/e1_o.bin,
/tmp/e1_core.bin and /tmp/e1_y.bin are stale after correcting GGUF
SSM-A and Q/K mapping. Error thresholds were NOT changed.

An independent corrected CPU oracle is generated with
scripts/block0_corrected_ref.py (NumPy, direct readonly GGUF mmap,
no CUDA, no torch, no legacy reference inputs). It independently
computes embedding/RMS, QKV, causal convolution, beta, gl, GDN
recurrent state, output projection and MLP for all 11 fixed token IDs.
Its output is the single binary /tmp/h27_block0_corrected_v2.bin.

Correct equations:

- GGUF blk.0.ssm_a already contains negative -exp(A_log), so
  gl = ssm_a * softplus(h @ ssm_alpha.T + ssm_dt.bias).
- The 16 Q/K groups are tiled to 48 V heads via hh % 16,
  not by consecutive repeated heads.
- The convolution starts with zero history, and GDN state persists
  across all 11 tokens. The oracle also regenerates core and final y.

CPU generation on the server:

    python3 /home/matteo/hermes-27/scripts/block0_corrected_ref.py --threads 2

CPU-only verification in the CUDA-linked P1 test:

    /home/matteo/hermes-27/build/test_p1_block0 oracle

Full GPU test ONLY after coordinator grants GPU:

    /home/matteo/hermes-27/build/test_p1_block0 all

The oracle-check subcommand and every GPU test mode first verify a
versioned binary header (magic H27B0V2!), model byte size, FNV1a64
of first/last 4096 checkpoint bytes, exact token IDs, payload size
and finiteness. Any missing, truncated, nonmatching or obsolete oracle
prints P1 BLOCK0 BLOCKED and returns 77 (nonzero). There is no
automatic fallback to E1, and no false PASS.

Format: 8-byte magic; uint64 model size; uint64 checkpoint fingerprint;
11 int32 IDs; then contiguous float32 arrays in this order:
qkv [11,10240], cv [11,10240], beta [11,48], gl [11,48],
o [11,48,128], core [11,5120], y [11,5120].

Environment HERMES_BLOCK0_ORACLE can override the file path for
separate scratch verification. The fingerprint protects against
accidental use of a different checkpoint but is not cryptographic.

CPU-only evidence from the current model (no GPU):

| output | max abs difference vs stale E1 |
|---|---:|
| qkv | 2.67e-05 |
| cv | 7.63e-06 |
| beta | 4.77e-07 |
| gl | 19.4512 |
| o | 2.02785 |
| core | 1.81134 |
| y | 3.30972 |

Seven arrays were finite, and the generated artifact contained
1,626,308 bytes. Python py_compile passed. nvcc -std=c++17 -c of
the test file in isolated scratch passed. No GPU STAGE3 or STAGE4
PASS is claimed until hardware execution is authorized.
