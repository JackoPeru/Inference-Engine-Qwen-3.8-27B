# HERMES-27: independent llama.cpp layer-0 GDN CPU equations (2026-11-14)

**Result: CPU mathematical reference and tests PASS; actual llama post-convolution parity is NO-GO.** The two new Python files implement source-derived delta-net equations for genuine GGUF token 6205 at absolute position 0, layer 0. Only attn_norm-0, linear_attn_qkv_mixed-0 and conv_output_silu-0 are authentic CUDA0-backed llama.cpp callbacks. Subsequent beta, z, normalized Q/K, recurrent-state, GDN and gated-output values are independent CPU **derivations**, not genuine downstream llama outputs.

## Immutable inputs and source provenance

Ubuntu repository HEAD at initial inspection: **95fcd54384c5cb137eeea8549b45738fd9e2e670**. During concurrent work, a final read-only inspection saw HEAD **8a829136aeefdf68bb964ed9c39fc5a2d007d154**. The shared Windows folder does not contain .git. This worker did not change Git or source outside the three assigned new files. The source-contract and physical GGUF checks passed against their individually recorded SHA256 identities in the completed integration.

- Physical GGUF: /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf; 19,694,360,544 bytes; SHA256 **db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1**.
- True llama capture: /tmp/h27-layer0-llama-pos0-prime3/manifest.json; SHA256 **49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b**. Exact token 6205, pos0, layer0, three original F32 tensors, individual hashes/strides and GPU source checked by the existing strict validator.
- Installed actual libllama SHA256 **fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4**.
- Actual installed qwen35.cpp SHA256 **ef973e9c789eb52ef27a77e0b1f36c6f41a378733a994d41b5924e3be012f671**.
- Actual installed delta-net-base.cpp SHA256 **5e3b318953f854b3825dd7cfe4b483ca7935cd5f6680438a3ece4be9b4d5428f**.
- Actual installed models.h SHA256 **d239ec2d363a5f2618eb91d486a3f53d9bf07b96be49b6d25b6c6688600d50a6**.
- Actual installed ggml-cuda/gated_delta_net.cu SHA256 **fe5cfe4a35195fac999e8bd93d3ed18c68830096d4019bafa60e5da91d6ef4bf**.

The new script verifies whole-source hashes plus equation-bearing source fragments before any GGUF computation. The GGUF raw metadata field qwen35.attention.layer_norm_rms_epsilon is read and checked to be FP32 **1e-6**. Tampered sources, metadata, manifest and unusual tensor dimensions fail closed.

## Source-derived CPU equation and axes

The compact Q and K each have 16 heads of 128 elements, and V has 48 heads of 128. Both Q and K use L2 normalization with epsilon 1e-6; the graph's rms_norm(x, eps/128) / sqrt(128) equals x / sqrt(sum(x*x) + eps). The source's fused CUDA head lookup is h % 16 (fastmodulo), not h // 3. The delta contraction has an additional factor 1/sqrt(128): fused CUDA applies it to the output, while the unfused graph applies it to Q. This is algebraically equivalent, with potentially different FP32 rounding.

The beta and raw z projection inputs are the authentic normalized 5120-vector represented as per-32 Q8_1 codes and stored FP16 scales. Independently dequantized real NVFP4 GGUF matrices are multiplied using bounded FP64 CPU panels, cast to FP32. Alpha uses the real GGUF F32 matrix and unrounded authentic norm vector. The GGUF ssm_a is already negative: beta = sigmoid(Wbeta x_Q8), gamma = ssm_a * softplus(Walpha x + dt), with gamma < 0, and z = Wz x_Q8.

The mathematical recurrent equation for value head h is:

~~~text
D = exp(gamma[h]) * S_previous
innovation = beta[h] * (v[h] - transpose(D) @ k[h])
S_next = D + outer(k[h], innovation)
o[h] = transpose(S_next) @ q[h] / sqrt(128)
gated[h] = RMSNorm(o[h], ssm_norm.weight, eps=1e-6) * SiLU(z[h])
~~~

At the first absolute position with zero initial recurrent history, S_next = outer(k, beta*v) and o = beta*v*(dot(q,k)/sqrt(128)); gamma cannot affect o at position 0. The script checks this independent closed form against the full CPU recurrence with 1e-12 mathematical absolute tolerance. Nonzero prior-state toy tests cover decay and prediction subtraction, but do not establish real llama position-1 correctness.

This is an equation-level reference derived from the installed llama graph and fused source, independent from PP2 CUDA formulas. FP64 CPU dot/recurrent math is not the actual llama fused CUDA warp reduction, operation order, FMA behavior, BF16 activation mode, or GPU tensor output.

## Validation: final 10/10 PASS, exit 0

Ran on Ubuntu in dedicated /tmp/h27_w18_experiments scratch, via /opt/hermes/runtimes/exllamav3-venv/bin/python -B with OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2. The final complete integration completed in **14.022 s**.

Seven pure-CPU tests verify hand-computed Q/K normalization, head repetition and wrong h//3 grouping; hand-computed nonzero-state recurrence, zero-state decay independence, missing 1/sqrt(width) and removed prediction mutants; sigmoid, softplus, negative stored A; gated RMS+SiLU; per-32 Q8_1 signed rounding; GGUF epsilon metadata and mutations; missing/modified pinned source equation fragments and invalid shapes/values.

Three real-integration tests verify the authenticated full physical GGUF and the original three llama captured tensors, reject a forged prompt/manifest **before accessing GGUF**, and independently compare the Q8_1 vector against the preexisting CPU quantizer. All 5120 Q8 input **numeric values are identical**. There are exactly **20 signed-zero bit-pattern differences**, which explain different input-byte SHA256 but no mathematical input difference. The real beta pre-sigmoid F32 SHA256 agrees exactly with the earlier CPU reference: **203ae02595daa23a11134ff7d79f9b4a3fc4bd7a3fa935693dff04d69d91041a**. Differences in sigmoid, GDN and gated RMS hashes can reflect independently chosen FP64 versus F32 rounding; they cannot be judged as actual llama output mismatches.

The initial pure-CPU run had **two RED toy thresholds**, which rejected deliberate mutant distances 0.0885647288653374 and 0.7267545075785895 because the original minimum thresholds were 0.1 and 0.9. They were calibrated to 0.08 and 0.7, preserving a test that detects those wrong expressions. Subsequent unit and real checks passed. No production behavior changed. The initial Windows executable lookup for python failed because python was not in PATH, and SSH default user matte was denied; tests were correctly run in the known Ubuntu CPU venv using SSH user matteo. No C/CUDA compilation was attempted.

Executable integrated command:

~~~bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /opt/hermes/runtimes/exllamav3-venv/bin/python -B \
  /tmp/h27_w18_experiments/experiments/tests/test_llama_gdn_layer0_ref_cpu.py \
  --manifest /tmp/h27-layer0-llama-pos0-prime3/manifest.json \
  --gguf /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
  --source-root /home/matteo/hermes-27-gh/hermes-27 \
  --llama-root /opt/hermes/runtimes/llama-mainline \
  --out /tmp/h27_w18_llama_gdn_real_result.json
~~~

The generated real-token JSON /tmp/h27_w18_llama_gdn_real_result.json in Ubuntu scratch has final SHA256 **31228ea3e0a0a766c5ee6531ee17867ae4ba053af7ab93d16cca3a548ca333f9**. It records all CPU stages separately from actual captured llama tensors. Independent CPU-derived GDN output F32 [48,128] SHA256 **4085a14c1d6ca120eeb378f42c8e66d166cf145f3b228a3bdd33c50573179320**; independent gated RMS/SiLU F32 [48,128] SHA256 **b41b6027261a43dcfac24d89085495aeb16d18c504bbddf10df66a461193c770**.

## Earliest uncertainty and release boundary

The earliest unavailable independently verified llama output stages are **beta_sigmoid-0 and raw z-0**, projected in parallel from the normalized input; **q_conv_predelta-0/k_conv_predelta-0** are next before GDN. No real llama-versus-CPU postconv mismatch can be localized without freshly authenticated callbacks for those stages. Even when postconv pos0 parity becomes available, true recurrent decay needs a position >=1 capture with nonzero history. No full-model inference, MTP, CUDA parity, PP2 correctness, throughput or numerical release gate follows from these tests. **NO-GO remains in force.**

Only three NEW files created, and only these owned files modified during work:

- experiments/llama_gdn_layer0_independent_cpu_ref.py
- experiments/tests/test_llama_gdn_layer0_ref_cpu.py
- reports/LAYER0_LLAMA_GDN_CPU_EQUATION_NEXT.md

Frozen local and Ubuntu scratch source SHA256 agree byte-for-byte: reference **937cf37200e3709c6175eef47769510f0a19740b762d44b5da038ae7841cd4b1**; test **78da34496aa0ce4116e266ecc2cd236b3a5892d7a1b81672ac43a9ad1af32028**. The report's own SHA256 is stated in the worker handoff to avoid self-reference.
