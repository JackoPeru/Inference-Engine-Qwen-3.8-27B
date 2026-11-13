// Compile with g++ only. Inspects one REAL GGUF NVFP4 tensor and (optionally)
// reconstructs / checks one REAL decode activation; never initializes CUDA.
#include "gemv_real_common.hpp"

int main(int argc, char** argv) {
  try {
    const auto options = gemv_real::parse(argc, argv, false);
    if (options.selftest) {
      gemv_real::selftest();
      return 0;
    }
    auto replay = gemv_real::load_weights(options);
    if (!options.inspect_weights)
      gemv_real::load_activation(replay, options);
    gemv_real::print_identity(replay);
    if (options.inspect_weights) {
      std::cout << "WEIGHTS_INSPECT PASS all UE4M3 scale bytes in selected real tensor\n";
      return 0;
    }
    const auto oracle = gemv_real::cpu_fp64_all(replay);
    gemv_real::print_activation_rounding(replay, oracle);
    double max_abs = 0.0, sumsq = 0.0, max_bound = 0.0;
    int max_row = -1;
    for (int row = 0; row < replay.F; ++row) {
      const double magnitude = std::fabs(oracle[row].dot);
      sumsq += oracle[row].dot * oracle[row].dot;
      if (magnitude > max_abs) { max_abs = magnitude; max_row = row; }
      max_bound = std::max(max_bound,
                           gemv_real::fp32_error_envelope(replay.H, oracle[row]));
    }
    std::cout << std::setprecision(10)
              << "CPU_FP64 PASS full_rows=" << oracle.size()
              << " ref_abs_max=" << max_abs << " row=" << max_row
              << " ref_rms=" << std::sqrt(sumsq / replay.F)
              << " conservative_fp32_envelope_max=" << max_bound << '\n';
    if (replay.reconstructed && !replay.direct_bf16_confirmed)
      std::cout << "CAPTURE_NOTE: BF16 reconstructed with RNE from d_g F32; "
                   "actual device d_gb was not independently captured\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL_OR_BLOCKED gemv_real_cpu: " << e.what() << '\n';
    return 1;
  }
}
