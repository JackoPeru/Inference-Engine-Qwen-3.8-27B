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
