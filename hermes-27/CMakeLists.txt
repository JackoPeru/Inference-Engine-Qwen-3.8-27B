cmake_minimum_required(VERSION 3.22)
project(hermes27 CXX CUDA)
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
  kernels/src/head_bf16_optimized.cu
  kernels/src/gdn_step.cu
  kernels/src/nvfp4_gemm_w4a4.cu
)
target_include_directories(hermes_kernels PUBLIC kernels/include
  ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include
  ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/tools/util/include)
set_target_properties(hermes_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(test_rmsnorm kernels/tests/test_rmsnorm.cu)
target_link_libraries(test_rmsnorm PRIVATE hermes_kernels)
add_test(NAME rmsnorm COMMAND test_rmsnorm)
add_executable(test_rope kernels/tests/test_rope.cu)
target_link_libraries(test_rope PRIVATE hermes_kernels)
add_test(NAME rope COMMAND test_rope)
add_executable(test_mlp kernels/tests/test_mlp.cu)
target_link_libraries(test_mlp PRIVATE hermes_kernels)
add_test(NAME mlp COMMAND test_mlp)
add_executable(test_attn kernels/tests/test_attn.cu)
target_link_libraries(test_attn PRIVATE hermes_kernels)
add_test(NAME attn COMMAND test_attn)
add_executable(test_nvfp4_gemv kernels/tests/test_nvfp4_gemv.cu)
target_link_libraries(test_nvfp4_gemv PRIVATE hermes_kernels)
add_test(NAME nvfp4_gemv COMMAND test_nvfp4_gemv)
add_executable(test_gdn_step kernels/tests/test_gdn_step.cu)
target_link_libraries(test_gdn_step PRIVATE hermes_kernels)
add_test(NAME gdn_step COMMAND test_gdn_step)
add_executable(test_nvfp4_w4a4 kernels/tests/test_nvfp4_w4a4.cu)
target_link_libraries(test_nvfp4_w4a4 PRIVATE hermes_kernels)
add_test(NAME nvfp4_w4a4 COMMAND test_nvfp4_w4a4)
add_executable(test_nvfp4_const kernels/tests/test_nvfp4_const.cu)
target_link_libraries(test_nvfp4_const PRIVATE hermes_kernels)
add_test(NAME nvfp4_const COMMAND test_nvfp4_const)
# test_nvfp4_layout removed: read-back path invalid (see order/struct probes instead)
add_executable(test_nvfp4_order kernels/tests/test_nvfp4_order.cu)
target_link_libraries(test_nvfp4_order PRIVATE hermes_kernels)
add_test(NAME nvfp4_order COMMAND test_nvfp4_order)
add_executable(test_p1_block0 kernels/tests/test_p1_block0.cu)
target_link_libraries(test_p1_block0 PRIVATE hermes_gguf hermes_kernels)
add_test(NAME p1_block0 COMMAND test_p1_block0)
add_executable(test_p1_block3 kernels/tests/test_p1_block3.cu)
target_link_libraries(test_p1_block3 PRIVATE hermes_gguf hermes_kernels)
add_test(NAME p1_block3 COMMAND test_p1_block3)
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
add_test(NAME gdn_block COMMAND test_gdn_block)
add_executable(test_model_fwd architecture/tests/test_model_fwd.cpp)
target_link_libraries(test_model_fwd PRIVATE hermes_arch hermes_gguf)
add_test(NAME model_fwd COMMAND test_model_fwd)
add_executable(test_mtp_accept architecture/tests/test_mtp_accept.cpp)
target_link_libraries(test_mtp_accept PRIVATE hermes_arch hermes_gguf)
add_test(NAME mtp_accept COMMAND test_mtp_accept)
add_executable(test_logits_dump architecture/tests/test_logits_dump.cpp)
target_link_libraries(test_logits_dump PRIVATE hermes_arch hermes_gguf)
# NOT in ctest: heavy full-prefill saver, run only explicitly in queue.
# P2 PP2 forward (distributed, GEMV preciso, single-copy raw GGUF residente)
add_library(hermes_pp2 STATIC
  distributed/pp2_fwd.cu
)
target_include_directories(hermes_pp2 PUBLIC distributed kernels/include model/include
  ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include
  ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/tools/util/include)
set_target_properties(hermes_pp2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hermes_pp2 PUBLIC hermes_gguf hermes_kernels)
add_executable(hermes_generate tools/hermes_generate.cpp)
target_link_libraries(hermes_generate PRIVATE hermes_pp2)
add_dependencies(hermes_generate gguf_dump)
add_executable(test_pp2_fwd distributed/test_pp2_fwd.cu)
target_link_libraries(test_pp2_fwd PRIVATE hermes_pp2 hermes_gguf hermes_kernels)
add_test(NAME pp2_fwd COMMAND test_pp2_fwd)
add_executable(test_pp2_p2 distributed/test_pp2_p2.cpp)
target_link_libraries(test_pp2_p2 PRIVATE hermes_pp2)
# Run explicitly: second prompt with the full checkpoint.
# P4 PP2 decode incrementale (stati su GPU, riuso upload P2)
add_executable(test_pp2_decode distributed/test_pp2_decode.cu)
target_link_libraries(test_pp2_decode PRIVATE hermes_pp2 hermes_gguf hermes_kernels)
add_test(NAME pp2_decode COMMAND test_pp2_decode)
# P5 PP2 batch concorrente (N richieste, stati separati, overlap pipeline)
find_package(Threads REQUIRED)
add_executable(test_pp2_batch distributed/test_pp2_batch.cu distributed/pp2_batch.cu)
target_link_libraries(test_pp2_batch PRIVATE hermes_pp2 hermes_gguf hermes_kernels Threads::Threads)
add_test(NAME pp2_batch COMMAND test_pp2_batch)

add_library(hermes_mtp STATIC distributed/mtp_gpu.cu)
set_target_properties(hermes_mtp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hermes_mtp PUBLIC hermes_pp2)
add_executable(test_mtp_gpu distributed/test_mtp_gpu.cu)
target_link_libraries(test_mtp_gpu PRIVATE hermes_mtp)
add_test(NAME mtp_gpu COMMAND test_mtp_gpu)

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
  message(STATUS "hermes_server disabled: cpp-httplib and nlohmann/json headers not found")
endif()
