cmake_minimum_required(VERSION 3.22)
project(hermes27 CXX CUDA)
find_package(CUDAToolkit REQUIRED)
option(HERMES_REQUIRE_SERVER "Fail configuration if HTTP serving dependencies are unavailable" OFF)
option(HERMES_BUILD_STREAM_TEST "Build isolated two-GPU async stream prototype regression test" OFF)
option(HERMES_BUILD_REAL_ASYNC_TEST "Build real-GGUF multiuser async PP2 correctness test (run only with exclusive GPUs)" OFF)
option(HERMES_BUILD_EXACT_PREFILL_BENCH "Build isolated BF16-exact NVFP4 prefill benchmark and CPU numeric gate" OFF)
option(HERMES_BUILD_MTP_GENERATOR "Build verified greedy MTP diagnostic CLI (not a speculative speedup path)" OFF)
option(HERMES_BUILD_W4A4_PROTOTYPE "Build unapproved CUTLASS NVFP4 activation-requantizing W4A4 prototype" OFF)
option(HERMES_BUILD_REAL_GEMV_ORACLE "Build real GGUF same-X/W Scalar/Warp4/Packed4 oracle (GPU execution manual only)" OFF)
option(HERMES_BUILD_LAYER0_LLAMA_GEMV_ORACLE "Build same-GGUF layer0 llama-capture GEMV differential (GPU execution manual only)" OFF)
option(HERMES_BUILD_REAL_PREFILL_TEST "Build real GGUF batched prefill state/cache/continuation gate (GPU execution manual only)" OFF)
option(HERMES_BUILD_CHUNK_PREFILL_TEST "Build experimental T65..8192 PP2 chunked prefill test (manual GPU execution only)" OFF)
option(HERMES_REGISTER_GPU_TESTS "Register CUDA tests in CTest (run ONLY during exclusive GPU maintenance)" OFF)
option(HERMES_REGISTER_HEAVY_CPU_TESTS "Register full-GGUF CPU reference tests (up to ~20min each)" OFF)
option(HERMES_REGISTER_EXTERNAL_FIXTURE_TESTS "Register CPU GDN test requiring /tmp/gdn_tiny.bin fixture" OFF)
set(HERMES_CUTLASS_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/cutlass" CACHE PATH
  "CUTLASS source checkout for experimental W4A4 prototype (not required by production)")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 120a)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
add_library(hermes_gguf STATIC
  model/src/gguf.cpp
  model/src/weights.cpp
)
target_include_directories(hermes_gguf PUBLIC model/include)
add_executable(gguf_dump tools/gguf_dump.cpp)
target_link_libraries(gguf_dump PRIVATE hermes_gguf)
enable_testing()
# These tests are host-only and never initialize a CUDA model. Keep the full
# checkpoint / multi-GPU gates opt-in and outside automatic CTest discovery.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME logit_comparator_cpu
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tools/compare_fp32_logits.py --self-test)
  add_test(NAME release_harness_cpu
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tests/reliability_selftest.py)
  add_test(NAME async_stream_contract_cpu
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/distributed/test_async_stream_contract.py)
  add_test(NAME prefill_integration_contract_cpu
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/distributed/test_prefill_integration_contract.py)
  set_tests_properties(logit_comparator_cpu release_harness_cpu async_stream_contract_cpu prefill_integration_contract_cpu
    PROPERTIES TIMEOUT 60)
endif()
add_executable(test_gguf_synth model/tests/test_gguf_synth.cpp)
target_link_libraries(test_gguf_synth PRIVATE hermes_gguf)
add_test(NAME gguf_synth COMMAND test_gguf_synth)
# kernels (SM120a, CUDA 12.8+/13.2)
add_library(hermes_kernels STATIC
  kernels/src/rmsnorm.cu
  kernels/src/rope.cu
  kernels/src/mlp.cu
  kernels/src/attn.cu
  kernels/src/nvfp4_gemv.cu
  kernels/src/nvfp4_gemv_warp_variants.cu
  # Exact GGUF -> BF16 temporary weights + FP32-accumulating Tensor Core GEMM.
  # Only selected when the PP2 experimental prefill flag is explicitly enabled.
  kernels/src/nvfp4_prefill_exact.cu
  kernels/src/head_bf16_optimized.cu
  kernels/src/gdn_step.cu
)
target_include_directories(hermes_kernels PUBLIC kernels/include)
if(HERMES_BUILD_W4A4_PROTOTYPE)
  if(NOT EXISTS "${HERMES_CUTLASS_ROOT}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "W4A4 prototype requires CUTLASS: set HERMES_CUTLASS_ROOT to a checkout containing include/cutlass/cutlass.h")
  endif()
  target_sources(hermes_kernels PRIVATE kernels/src/nvfp4_gemm_w4a4.cu)
  target_include_directories(hermes_kernels PRIVATE
    "${HERMES_CUTLASS_ROOT}/include" "${HERMES_CUTLASS_ROOT}/tools/util/include")
else()
  # PP2 still exposes an old upload-only CUTLASS timing probe. Keep its ABI
  # linkable but make all W4A4 entrypoints return unsupported / zero sizes.
  # The production GGUF GEMV and BF16-exact prefill are unaffected.
  target_sources(hermes_kernels PRIVATE kernels/src/nvfp4_gemm_w4a4_unavailable.cpp)
endif()
target_link_libraries(hermes_kernels PUBLIC CUDA::cublas)
set_target_properties(hermes_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(HERMES_BUILD_REAL_GEMV_ORACLE)
  add_executable(gemv_real_cpu kernels/tests/gemv_real_cpu.cpp)
  target_include_directories(gemv_real_cpu PRIVATE kernels/tests)
  target_link_libraries(gemv_real_cpu PRIVATE hermes_gguf)
  add_executable(gemv_real_q8_cpu kernels/tests/gemv_real_q8_cpu.cpp)
  target_include_directories(gemv_real_q8_cpu PRIVATE kernels/tests)
  target_link_libraries(gemv_real_q8_cpu PRIVATE hermes_gguf)
  add_test(NAME gemv_real_cpu_selftest COMMAND gemv_real_cpu --selftest)
  add_test(NAME gemv_real_q8_cpu_selftest COMMAND gemv_real_q8_cpu --selftest)
  add_executable(gemv_real_compare kernels/tests/gemv_real_compare.cu)
  target_include_directories(gemv_real_compare PRIVATE kernels/tests)
  target_link_libraries(gemv_real_compare PRIVATE hermes_gguf hermes_kernels)
  # Do NOT add gemv_real_compare to CTest; the real-checkpoint workload
  # is only permitted with both GPUs exclusively reserved.
endif()
if(HERMES_BUILD_LAYER0_LLAMA_GEMV_ORACLE)
  add_executable(gemv_layer0_compare kernels/tests/gemv_layer0_compare.cu)
  target_include_directories(gemv_layer0_compare PRIVATE kernels/tests)
  target_link_libraries(gemv_layer0_compare PRIVATE hermes_gguf hermes_kernels)
  # Requires an independently authenticated llama.cpp capture and explicit
  # GPU exclusivity. Never register this full-checkpoint test in CTest.
endif()
add_executable(test_rmsnorm kernels/tests/test_rmsnorm.cu)
target_link_libraries(test_rmsnorm PRIVATE hermes_kernels)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME rmsnorm COMMAND test_rmsnorm)
endif()
add_executable(test_rope kernels/tests/test_rope.cu)
target_link_libraries(test_rope PRIVATE hermes_kernels)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME rope COMMAND test_rope)
endif()
add_executable(test_mlp kernels/tests/test_mlp.cu)
target_link_libraries(test_mlp PRIVATE hermes_kernels)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME mlp COMMAND test_mlp)
endif()
add_executable(test_attn kernels/tests/test_attn.cu)
target_link_libraries(test_attn PRIVATE hermes_kernels)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME attn COMMAND test_attn)
endif()
add_executable(test_nvfp4_gemv kernels/tests/test_nvfp4_gemv.cu)
target_link_libraries(test_nvfp4_gemv PRIVATE hermes_kernels)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME nvfp4_gemv COMMAND test_nvfp4_gemv)
endif()
add_executable(test_gdn_step kernels/tests/test_gdn_step.cu)
target_link_libraries(test_gdn_step PRIVATE hermes_kernels)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME gdn_step COMMAND test_gdn_step)
endif()
if(HERMES_BUILD_W4A4_PROTOTYPE)
  add_executable(test_nvfp4_w4a4 kernels/tests/test_nvfp4_w4a4.cu)
  target_link_libraries(test_nvfp4_w4a4 PRIVATE hermes_kernels)
  if(HERMES_REGISTER_GPU_TESTS)
    add_test(NAME nvfp4_w4a4 COMMAND test_nvfp4_w4a4)
  endif()
  add_executable(test_nvfp4_const kernels/tests/test_nvfp4_const.cu)
  target_link_libraries(test_nvfp4_const PRIVATE hermes_kernels)
  if(HERMES_REGISTER_GPU_TESTS)
    add_test(NAME nvfp4_const COMMAND test_nvfp4_const)
  endif()
  add_executable(test_nvfp4_order kernels/tests/test_nvfp4_order.cu)
  target_link_libraries(test_nvfp4_order PRIVATE hermes_kernels)
  if(HERMES_REGISTER_GPU_TESTS)
    add_test(NAME nvfp4_order COMMAND test_nvfp4_order)
  endif()
endif()
if(HERMES_BUILD_EXACT_PREFILL_BENCH)
  add_executable(prefill_exact_bench
    kernels/tests/prefill_exact_bench.cu)
  target_include_directories(prefill_exact_bench PRIVATE kernels/include model/include)
  target_link_libraries(prefill_exact_bench PRIVATE hermes_gguf hermes_kernels CUDA::cublas)
  add_test(NAME prefill_exact_cpu COMMAND prefill_exact_bench --cpu-only)
endif()
# test_nvfp4_layout removed: invalid read-back; W4A4 const/order/struct
# probes are guarded with the unapproved CUTLASS prototype above.
if(HERMES_BUILD_W4A4_PROTOTYPE)
  # These tests invoke CUTLASS W4A4 conversion. Omitting the backend must
  # not turn an unsupported prototype into a false quality-gate PASS.
  add_executable(test_p1_block0 kernels/tests/test_p1_block0.cu)
  target_link_libraries(test_p1_block0 PRIVATE hermes_gguf hermes_kernels)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME p1_block0 COMMAND test_p1_block0)
  set_tests_properties(p1_block0 PROPERTIES SKIP_RETURN_CODE 77)
endif()
  add_executable(test_p1_block3 kernels/tests/test_p1_block3.cu)
  target_link_libraries(test_p1_block3 PRIVATE hermes_gguf hermes_kernels)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME p1_block3 COMMAND test_p1_block3)
endif()
else()
  message(STATUS "CUTLASS W4A4 P1 prototype tests disabled: optional backend absent")
endif()
# test_nvfp4_vsref removed: superseded by w4a4 dequant-ref comparison (tighter)
# architecture (CPU fp32 reference; CUDA step kernels to follow)
add_library(hermes_arch STATIC
  architecture/src/gdn_block.cpp
  architecture/src/model_fwd.cpp
)
target_link_libraries(hermes_arch PUBLIC hermes_gguf)
target_include_directories(hermes_arch PUBLIC architecture/include)
add_executable(test_gdn_block architecture/tests/test_gdn_block.cpp)
target_link_libraries(test_gdn_block PRIVATE hermes_arch)
if(HERMES_REGISTER_EXTERNAL_FIXTURE_TESTS)
  if(EXISTS "/tmp/gdn_tiny.bin")
    add_test(NAME gdn_block COMMAND test_gdn_block)
  else()
    message(WARNING "Skipping GDN external-fixture test: /tmp/gdn_tiny.bin not found")
  endif()
endif()
add_executable(test_model_fwd architecture/tests/test_model_fwd.cpp)
target_link_libraries(test_model_fwd PRIVATE hermes_arch hermes_gguf)
add_executable(test_mtp_accept architecture/tests/test_mtp_accept.cpp)
target_link_libraries(test_mtp_accept PRIVATE hermes_arch hermes_gguf)
if(HERMES_REGISTER_HEAVY_CPU_TESTS)
  add_test(NAME model_fwd COMMAND test_model_fwd)
  add_test(NAME mtp_accept COMMAND test_mtp_accept)
  set_tests_properties(model_fwd mtp_accept PROPERTIES TIMEOUT 1800 LABELS "slow;real-gguf")
endif()
add_executable(test_logits_dump architecture/tests/test_logits_dump.cpp)
target_link_libraries(test_logits_dump PRIVATE hermes_arch hermes_gguf)
# NOT in ctest: heavy full-prefill saver, run only explicitly in queue.
# P2 PP2 forward (distributed, GEMV preciso, single-copy raw GGUF residente)
add_library(hermes_pp2 STATIC
  distributed/pp2_fwd.cu
)
target_include_directories(hermes_pp2 PUBLIC distributed kernels/include model/include)
set_target_properties(hermes_pp2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hermes_pp2 PUBLIC hermes_gguf hermes_kernels)
target_compile_definitions(hermes_pp2 PRIVATE
  HERMES_HAVE_W4A4=$<BOOL:${HERMES_BUILD_W4A4_PROTOTYPE}>)
if(HERMES_BUILD_REAL_PREFILL_TEST)
  add_executable(test_prefill_integration_real distributed/test_prefill_integration_real.cu)
  target_include_directories(test_prefill_integration_real PRIVATE distributed)
  target_link_libraries(test_prefill_integration_real PRIVATE
    hermes_pp2 hermes_kernels hermes_gguf CUDA::cublas Threads::Threads)
  # Not registered with CTest: an explicit prime-owned GPU window is required.
endif()
if(HERMES_BUILD_CHUNK_PREFILL_TEST)
  add_executable(test_prefill_chunk_real distributed/test_prefill_chunk_real.cu)
  target_include_directories(test_prefill_chunk_real PRIVATE distributed)
  target_link_libraries(test_prefill_chunk_real PRIVATE
    hermes_pp2 hermes_kernels hermes_gguf CUDA::cublas Threads::Threads)
  # Only offline CPU checks may run in CTest without GPU exclusivity.
  if(Python3_Interpreter_FOUND)
    add_test(NAME prefill_chunk_cpu COMMAND
      ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/distributed/test_prefill_chunk_cpu.py)
    set_tests_properties(prefill_chunk_cpu PROPERTIES TIMEOUT 60)
  endif()
endif()
add_executable(hermes_generate tools/hermes_generate.cpp)
target_link_libraries(hermes_generate PRIVATE hermes_pp2)
add_dependencies(hermes_generate gguf_dump)
if(Python3_Interpreter_FOUND)
  add_test(NAME generate_batched_args_cpu COMMAND
    ${Python3_EXECUTABLE} -B ${CMAKE_SOURCE_DIR}/tools/test_generate_args_cpu.py
    $<TARGET_FILE:hermes_generate>)
  set_tests_properties(generate_batched_args_cpu PROPERTIES TIMEOUT 60)
endif()
add_executable(test_pp2_fwd distributed/test_pp2_fwd.cu)
target_link_libraries(test_pp2_fwd PRIVATE hermes_pp2 hermes_gguf hermes_kernels)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME pp2_fwd COMMAND test_pp2_fwd)
endif()
add_executable(test_pp2_p2 distributed/test_pp2_p2.cpp)
target_link_libraries(test_pp2_p2 PRIVATE hermes_pp2)
# Run explicitly: second prompt with the full checkpoint.
# P4 PP2 decode incrementale (stati su GPU, riuso upload P2)
add_executable(test_pp2_decode distributed/test_pp2_decode.cu)
target_link_libraries(test_pp2_decode PRIVATE hermes_pp2 hermes_gguf hermes_kernels)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME pp2_decode COMMAND test_pp2_decode)
endif()
# P5 PP2 batch concorrente (N richieste, stati separati, overlap pipeline)
find_package(Threads REQUIRED)
if(HERMES_BUILD_STREAM_TEST)
  add_executable(test_batch_streams_synthetic
    distributed/pp2_batch_streams_prototype.cu
    distributed/test_batch_streams_synthetic.cu)
  target_include_directories(test_batch_streams_synthetic PRIVATE distributed)
  target_link_libraries(test_batch_streams_synthetic PRIVATE Threads::Threads)
  if(HERMES_REGISTER_GPU_TESTS)
    add_test(NAME batch_streams_synthetic COMMAND test_batch_streams_synthetic)
  endif()
endif()
add_executable(test_pp2_batch distributed/test_pp2_batch.cu distributed/pp2_batch.cu)
target_link_libraries(test_pp2_batch PRIVATE hermes_pp2 hermes_gguf hermes_kernels Threads::Threads)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME pp2_batch COMMAND test_pp2_batch)
endif()
if(HERMES_BUILD_REAL_ASYNC_TEST)
  add_executable(test_async_multiuser distributed/test_async_multiuser.cu)
  target_include_directories(test_async_multiuser PRIVATE distributed)
  target_link_libraries(test_async_multiuser PRIVATE hermes_pp2 hermes_kernels Threads::Threads)
  # Intentionally NOT an automatic ctest: this launches full GGUF on both GPUs.
  # Run manually only after GPU exclusivity and checkpoint verification.
endif()

add_library(hermes_mtp STATIC distributed/mtp_gpu.cu)
set_target_properties(hermes_mtp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hermes_mtp PUBLIC hermes_pp2)
add_executable(test_mtp_gpu distributed/test_mtp_gpu.cu)
target_link_libraries(test_mtp_gpu PRIVATE hermes_mtp)
if(HERMES_REGISTER_GPU_TESTS)
  add_test(NAME mtp_gpu COMMAND test_mtp_gpu)
endif()
if(HERMES_BUILD_MTP_GENERATOR)
  add_executable(hermes_mtp_generate tools/hermes_mtp_generate.cpp)
  target_include_directories(hermes_mtp_generate PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(hermes_mtp_generate PRIVATE hermes_mtp hermes_pp2 CUDA::cudart)
endif()

# Native HTTP serving; uses generic installed headers, never another inference runtime.
find_path(HERMES_HTTP_INCLUDE_DIR httplib.h
  HINTS /opt/hermes/src/llama-cpp-turboquant/vendor/cpp-httplib)
find_path(HERMES_JSON_INCLUDE_DIR nlohmann/json.hpp
  HINTS /opt/hermes/src/llama-cpp-turboquant/vendor)
if(HERMES_HTTP_INCLUDE_DIR AND HERMES_JSON_INCLUDE_DIR)
  add_executable(hermes_server serving/server.cpp distributed/pp2_batch.cu
    ${HERMES_HTTP_INCLUDE_DIR}/httplib.cpp)
  target_include_directories(hermes_server PRIVATE
    ${HERMES_HTTP_INCLUDE_DIR} ${HERMES_JSON_INCLUDE_DIR} distributed)
  target_link_libraries(hermes_server PRIVATE hermes_pp2 hermes_gguf hermes_kernels Threads::Threads)
else()
  if(HERMES_REQUIRE_SERVER)
    message(FATAL_ERROR "HERMES_REQUIRE_SERVER=ON but cpp-httplib and/or nlohmann/json headers were not found")
  endif()
  message(STATUS "hermes_server disabled: cpp-httplib and nlohmann/json headers not found")
endif()
