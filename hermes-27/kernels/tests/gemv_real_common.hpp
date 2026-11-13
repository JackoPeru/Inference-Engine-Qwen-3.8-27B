#pragma once
// CPU-side, standalone real-checkpoint replay support. Deliberately independent
// NVFP4 arithmetic; only GGUF metadata parsing is shared with the model loader.
#include "hermes/gguf.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gemv_real {

constexpr int kL26Vdd = 6144;
constexpr int kL26Hid = 5120;
// HERMES_L26 decode appends d_g[VDD], d_core[HID], d_o[HV*VHD], d_x[HID], F32.
constexpr std::uint64_t kL26RecordBytes =
    std::uint64_t(kL26Vdd + kL26Hid + kL26Vdd + kL26Hid) * 4;
constexpr int kRepeats = 10;

struct Arguments {
  std::string gguf;
  std::string tensor = "blk.26.ssm_out.weight";
  std::string bf16_path;
  std::string l26_path;
  std::string confirm_bf16_path;
  std::string original_f32_path;
  std::uint64_t record = 0;
  int device = 0;
  bool selftest = false;
  bool inspect_weights = false;
};

[[noreturn]] inline void error(const std::string& why) {
  throw std::runtime_error(why);
}

inline std::uint64_t parse_nonnegative(const std::string& text,
                                       const char* option) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    error(std::string(option) + " requires a nonnegative decimal integer");
  std::size_t used = 0;
  std::uint64_t number = 0;
  try { number = std::stoull(text, &used); }
  catch (const std::exception&) { error(std::string(option) + " integer overflow"); }
  if (used != text.size()) error(std::string(option) + " invalid integer");
  return number;
}

inline Arguments parse(int argc, char** argv, bool allow_gpu) {
  Arguments a;
  bool record_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string flag(argv[i]);
    if (flag == "--selftest") a.selftest = true;
    else if (flag == "--inspect-weights") a.inspect_weights = true;
    else {
      if (i + 1 >= argc) error("missing value for " + flag);
      const std::string value(argv[++i]);
      if (flag == "--gguf") a.gguf = value;
      else if (flag == "--tensor") a.tensor = value;
      else if (flag == "--activation-bf16") a.bf16_path = value;
      else if (flag == "--l26-dec") a.l26_path = value;
      else if (flag == "--confirm-bf16") a.confirm_bf16_path = value;
      else if (flag == "--original-f32") a.original_f32_path = value;
      else if (flag == "--record") {
        a.record = parse_nonnegative(value, "--record");
        record_set = true;
      } else if (flag == "--device" && allow_gpu) {
        const auto d = parse_nonnegative(value, "--device");
        if (d > 127) error("--device out of range");
        a.device = static_cast<int>(d);
      } else error("unknown option " + flag);
    }
  }
  if (a.selftest) {
    if (argc != 2) error("--selftest cannot be combined with other options");
    return a;
  }
  if (a.gguf.empty() || a.tensor.empty()) error("--gguf and --tensor are required");
  if (a.inspect_weights) {
    if (!a.bf16_path.empty() || !a.l26_path.empty() ||
        !a.confirm_bf16_path.empty() || !a.original_f32_path.empty() ||
        record_set || a.device != 0)
      error("--inspect-weights accepts only --gguf and --tensor");
    return a;
  }
  if (a.bf16_path.empty() == a.l26_path.empty())
    error("provide exactly one of --activation-bf16 or --l26-dec");
  if (record_set && a.l26_path.empty()) error("--record requires --l26-dec");
  if (!a.confirm_bf16_path.empty() && a.l26_path.empty())
    error("--confirm-bf16 requires --l26-dec");
  if (!a.original_f32_path.empty() && a.bf16_path.empty())
    error("--original-f32 requires --activation-bf16");
  if (!record_set && !a.l26_path.empty())
    error("--l26-dec requires explicit --record (avoid implicit/stale input)");
  if (!a.l26_path.empty() && a.tensor != "blk.26.ssm_out.weight")
    error("HERMES_L26 d_g is valid only for blk.26.ssm_out.weight");
  if (a.inspect_weights && allow_gpu) error("unexpected inspect mode");
  return a;
}

inline std::uint64_t checked_size(const std::string& path) {
  std::error_code ec;
  const auto n = std::filesystem::file_size(path, ec);
  if (ec) error("file size unavailable: " + path + ": " + ec.message());
  return n;
}

inline void read_exact(std::ifstream& file, void* dst, std::size_t bytes,
                       const std::string& label) {
  if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
    error("read size exceeds stream limit: " + label);
  file.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (!file || file.gcount() != static_cast<std::streamsize>(bytes))
    error("short read for " + label);
}

inline float bf16_float(std::uint16_t bits) {
  return std::bit_cast<float>(std::uint32_t(bits) << 16);
}

// CUDA __nv_bfloat16(float) uses nearest-even for finite F32 conversion.
// Reject nonfinite floats before conversion so the NaN payload policy is moot.
inline std::uint16_t f32_bf16_rne(std::uint32_t bits) {
  if ((bits & 0x7f800000u) == 0x7f800000u)
    error("nonfinite F32 in HERMES_L26 capture");
  return static_cast<std::uint16_t>(
      (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline double ue4m3_raw(std::uint8_t scale) {
  if (scale >= 128) error("GGUF UE4M3 scale high bit set; reject ambiguous decoding");
  if (scale == 0 || scale == 0x7f) return 0;
  const int exponent = (scale >> 3) & 15;
  const int mantissa = scale & 7;
  return exponent == 0 ? std::ldexp(double(mantissa), -9)
                       : std::ldexp(double(8 + mantissa), exponent - 10);
}

inline double e2m1(std::uint8_t q) {
  static constexpr double values[16] = {
      0., .5, 1., 1.5, 2., 3., 4., 6.,
      0., -.5, -1., -1.5, -2., -3., -4., -6.};
  return values[q & 15u];
}

struct Replay {
  std::string gguf;
  std::string tensor;
  int H = 0, F = 0;
  std::uint64_t weight_offset = 0;
  std::uint64_t model_bytes = 0;
  std::vector<std::uint8_t> weights;
  std::vector<std::uint16_t> x;
  std::vector<float> original_f32;
  bool reconstructed = false;
  bool direct_bf16_confirmed = false;
};

inline Replay load_weights(const Arguments& a) {
  if (std::endian::native != std::endian::little)
    error("GGUF/NVFP4 oracle requires little-endian host");
  const auto file_bytes = checked_size(a.gguf);
  hermes::GgufFile g;
  g.open(a.gguf);
  const auto& t = g.find(a.tensor);
  if (t.type != hermes::GgmlType::NVFP4 || t.n_dims != 2 ||
      t.ne[0] == 0 || t.ne[0] % 64 || t.ne[1] == 0 ||
      t.ne[0] > std::uint64_t(std::numeric_limits<int>::max()) ||
      t.ne[1] > std::uint64_t(std::numeric_limits<int>::max()))
    error("selected tensor must be 2D NVFP4 [F,H], H multiple of 64");
  const auto nbytes = (t.ne[0] / 64) * 36 * t.ne[1];
  // Deliberate bound: prevent a wrong tensor name from exhausting RAM.
  if (nbytes != t.n_bytes || nbytes > 256u * 1024u * 1024u)
    error("NVFP4 weight byte-size invalid or exceeds 256 MiB replay limit");
  const auto start = g.data_offset + t.offset;
  if (start > file_bytes || nbytes > file_bytes - start)
    error("NVFP4 tensor lies outside GGUF file");
  std::ifstream stream(a.gguf, std::ios::binary);
  if (!stream) error("could not open GGUF tensor bytes");
  stream.seekg(static_cast<std::streamoff>(start));
  if (!stream) error("GGUF tensor seek failed");
  Replay r;
  r.gguf = a.gguf;
  r.tensor = a.tensor;
  r.H = static_cast<int>(t.ne[0]);
  r.F = static_cast<int>(t.ne[1]);
  r.model_bytes = file_bytes;
  r.weight_offset = start;
  r.weights.resize(static_cast<std::size_t>(nbytes));
  read_exact(stream, r.weights.data(), r.weights.size(), "real GGUF NVFP4 tensor");
  const int blocks = r.H / 64;
  for (int row = 0; row < r.F; ++row)
    for (int b = 0; b < blocks; ++b)
      for (int s = 0; s < 4; ++s)
        (void)ue4m3_raw(r.weights[(static_cast<std::size_t>(row) * blocks + b) * 36 + s]);
  if (!a.l26_path.empty() && (r.H != kL26Vdd || r.F != kL26Hid))
    error("layer26 HERMES_L26 capture expects H=6144 and F=5120");
  return r;
}

inline void load_activation(Replay& r, const Arguments& a) {
  r.x.resize(r.H);
  if (!a.bf16_path.empty()) {
    if (checked_size(a.bf16_path) != std::uint64_t(r.H) * 2)
      error("raw BF16 activation length must be exactly H*2 bytes");
    std::ifstream f(a.bf16_path, std::ios::binary);
    if (!f) error("cannot open BF16 input");
    read_exact(f, r.x.data(), r.x.size() * 2, "raw BF16 activation");
    if (!a.original_f32_path.empty()) {
      if (checked_size(a.original_f32_path) != std::uint64_t(r.H) * 4)
        error("--original-f32 must contain exactly H*4 bytes");
      std::ifstream source(a.original_f32_path, std::ios::binary);
      if (!source) error("cannot open pre-BF16 F32 input");
      std::vector<std::uint32_t> bits(r.H);
      read_exact(source, bits.data(), bits.size() * 4, "original FP32 activation");
      r.original_f32.resize(r.H);
      for (int k = 0; k < r.H; ++k) {
        if (f32_bf16_rne(bits[k]) != r.x[k])
          error("original F32 rounded BF16 differs from supplied actual BF16 input");
        r.original_f32[k] = std::bit_cast<float>(bits[k]);
      }
      r.direct_bf16_confirmed = true;
    }
  } else {
    const auto size = checked_size(a.l26_path);
    if (size == 0 || size % kL26RecordBytes)
      error("HERMES_L26 log length is not a positive multiple of 90112 bytes");
    if (a.record >= size / kL26RecordBytes)
      error("requested HERMES_L26 record exceeds available complete records");
    if (a.record > std::uint64_t(std::numeric_limits<std::streamoff>::max()) /
                       kL26RecordBytes)
      error("capture record offset overflow");
    std::ifstream f(a.l26_path, std::ios::binary);
    if (!f) error("cannot open HERMES_L26 file");
    f.seekg(static_cast<std::streamoff>(a.record * kL26RecordBytes));
    if (!f) error("could not seek requested HERMES_L26 record");
    std::vector<std::uint32_t> f32(r.H);
    read_exact(f, f32.data(), f32.size() * 4, "HERMES_L26 d_g F32");
    r.original_f32.resize(r.H);
    for (int k = 0; k < r.H; ++k) {
      r.x[k] = f32_bf16_rne(f32[k]);
      r.original_f32[k] = std::bit_cast<float>(f32[k]);
    }
    r.reconstructed = true;
    if (!a.confirm_bf16_path.empty()) {
      if (checked_size(a.confirm_bf16_path) != std::uint64_t(r.H) * 2)
        error("--confirm-bf16 must contain exactly H*2 bytes");
      std::ifstream confirm(a.confirm_bf16_path, std::ios::binary);
      if (!confirm) error("cannot open direct BF16 confirmation");
      std::vector<std::uint16_t> actual(r.H);
      read_exact(confirm, actual.data(), actual.size() * 2, "BF16 confirmation");
      if (actual != r.x)
        error("reconstructed BF16 differs from actual captured BF16: abort replay");
      r.direct_bf16_confirmed = true;
    }
  }
  for (std::uint16_t b : r.x)
    if (!std::isfinite(bf16_float(b))) error("BF16 activation contains NaN/Inf");
}

struct ReferenceRow { double dot = 0.; double sumabs = 0.; };

inline ReferenceRow cpu_fp64_row(const Replay& r, int row,
                               bool original_f32 = false) {
  if (original_f32 && r.original_f32.size() != r.x.size())
    error("FP64 original activation requested without real F32 capture");
  const int blocks = r.H / 64;
  const std::uint8_t* w = r.weights.data() +
      static_cast<std::size_t>(row) * blocks * 36;
  ReferenceRow out;
  for (int b = 0; b < blocks; ++b)
    for (int s = 0; s < 4; ++s) {
      const double scale = ue4m3_raw(w[b * 36 + s]);
      for (int j = 0; j < 8; ++j) {
        const auto q = w[b * 36 + 4 + s * 8 + j];
        const int k = b * 64 + s * 16 + j;
        const double x0 = original_f32 ? double(r.original_f32[k])
                                       : double(bf16_float(r.x[k]));
        const double x1 = original_f32 ? double(r.original_f32[k + 8])
                                       : double(bf16_float(r.x[k + 8]));
        const double p0 = x0 * scale * e2m1(q);
        const double p1 = x1 * scale * e2m1(q >> 4);
        out.dot += p0;
        out.dot += p1;
        out.sumabs += std::fabs(p0) + std::fabs(p1);
      }
    }
  if (!std::isfinite(out.dot) || !std::isfinite(out.sumabs) ||
      !std::isfinite(static_cast<float>(out.dot)))
    error("nonfinite FP64 oracle result");
  return out;
}

inline std::vector<ReferenceRow> cpu_fp64_all(const Replay& r,
                                            bool original_f32 = false) {
  std::vector<ReferenceRow> out(r.F);
  for (int row = 0; row < r.F; ++row)
    out[row] = cpu_fp64_row(r, row, original_f32);
  return out;
}

inline void print_activation_rounding(const Replay& r,
    const std::vector<ReferenceRow>& bf16_reference) {
  if (r.original_f32.empty()) {
    std::cout << "FP32_VS_BF16_DOT unavailable: original pre-cast F32 input not provided\n";
    return;
  }
  const auto fp32_reference = cpu_fp64_all(r, true);
  double max_abs = 0., sum_sq = 0., sum_bf16_sq = 0.;
  int worst_row = -1;
  for (int row = 0; row < r.F; ++row) {
    const double error = fp32_reference[row].dot - bf16_reference[row].dot;
    const double absolute = std::fabs(error);
    if (absolute > max_abs) { max_abs = absolute; worst_row = row; }
    sum_sq += error * error;
    sum_bf16_sq += bf16_reference[row].dot * bf16_reference[row].dot;
  }
  const double rmse = std::sqrt(sum_sq / r.F);
  const double baseline_rms = std::sqrt(sum_bf16_sq / r.F);
  std::cout << std::setprecision(10)
            << "FP32_VS_BF16_DOT CPU_FP64_full_rows=" << r.F
            << " max_abs=" << max_abs << " rmse=" << rmse
            << " rel_rmse=" << rmse / std::max(baseline_rms, 1.e-30)
            << " worst_row=" << worst_row
            << " bf16_dot_rms=" << baseline_rms << '\n';
}

inline double ulp_float(double value) {
  const float near = static_cast<float>(value);
  const float hi = std::nextafter(near, std::numeric_limits<float>::infinity());
  const float lo = std::nextafter(near, -std::numeric_limits<float>::infinity());
  const double up = std::isfinite(hi) ? double(hi) - double(near) : 0.;
  const double down = std::isfinite(lo) ? double(near) - double(lo) : 0.;
  const double gap = std::max(up, down);
  if (!(gap > 0.) || !std::isfinite(gap)) error("FP32 ULP undefined");
  return gap;
}

// Conservative FP32 dot bound for these specific reduction shapes. Useful as
// an error diagnosis, not proof that model-level logits or quality are correct.
inline double fp32_error_envelope(int H, const ReferenceRow& row) {
  const std::uint64_t B = static_cast<std::uint64_t>(H) / 64;
  const std::uint64_t scalar_n = 3 * 64 * ((B + 255) / 256) + 8;
  const std::uint64_t packed_n = 16 * ((B + 3) / 4) + 5;
  const auto n = std::max(scalar_n, packed_n);
  const double roundoff = std::ldexp(1., -24);
  if (double(n) * roundoff >= 1.) error("FP32 bound invalid for huge H");
  const double gamma = double(n) * roundoff / (1. - double(n) * roundoff);
  return gamma * row.sumabs + 2. * ulp_float(row.dot);
}

inline std::uint64_t fnv1a(const void* buffer, std::size_t size) {
  const auto* data = static_cast<const std::uint8_t*>(buffer);
  std::uint64_t h = 14695981039346656037ull;
  for (std::size_t i = 0; i < size; ++i) {
    h ^= data[i];
    h *= 1099511628211ull;
  }
  return h; // diagnostic only, not a cryptographic checkpoint identity
}

inline void print_identity(const Replay& r) {
  std::cout << "REAL_GGUF tensor=" << r.tensor << " H=" << r.H << " F=" << r.F
            << " model_bytes=" << r.model_bytes
            << " offset=" << r.weight_offset
            << " weight_bytes=" << r.weights.size()
            << " weight_fnv64=" << std::hex << fnv1a(r.weights.data(), r.weights.size())
            << std::dec << '\n';
  if (!r.x.empty())
    std::cout << "ACTIVATION input=" << (r.reconstructed ? "l26-f32-rne" : "raw-bf16")
              << " bytes=" << r.x.size() * 2
              << " bf16_fnv64=" << std::hex << fnv1a(r.x.data(), r.x.size() * 2)
              << std::dec << " capture_confirmed=" << (r.reconstructed
                  ? (r.direct_bf16_confirmed ? "yes" : "no-reconstructed")
                  : (r.direct_bf16_confirmed ? "yes-f32-verified" : "provided-raw"))
              << '\n';
  if (!r.original_f32.empty())
    std::cout << "ACTIVATION_ORIGINAL_F32 bytes=" << r.original_f32.size() * 4
              << " f32_fnv64=" << std::hex
              << fnv1a(r.original_f32.data(), r.original_f32.size() * 4)
              << std::dec << '\n';
}

inline void selftest() {
  const struct { std::uint32_t input; std::uint16_t bf16; } conversions[] = {
      {0x3f800000u, 0x3f80u}, {0x3f808000u, 0x3f80u},
      {0x3f818000u, 0x3f82u}, {0xbf808000u, 0xbf80u},
      {0x00000000u, 0x0000u}, {0x80000000u, 0x8000u}};
  for (auto c : conversions)
    if (f32_bf16_rne(c.input) != c.bf16) error("F32->BF16 RNE selftest failed");
  if (ue4m3_raw(0x38) != 1. || ue4m3_raw(0x40) != 2. ||
      ue4m3_raw(0x7f) != 0. || ue4m3_raw(0x01) != std::ldexp(1., -9) ||
      e2m1(7) != 6. || e2m1(15) != -6. || e2m1(8) != 0.)
    error("NVFP4 independent dequant selftest failed");
  bool bad_scale_rejected = false, bad_activation_rejected = false;
  try { (void)ue4m3_raw(0x80); }
  catch (const std::runtime_error&) { bad_scale_rejected = true; }
  try { (void)f32_bf16_rne(0x7f800000u); }
  catch (const std::runtime_error&) { bad_activation_rejected = true; }
  if (!bad_scale_rejected || !bad_activation_rejected)
    error("noncanonical scale or nonfinite activation accepted");
  Replay r;
  r.H = 64;
  r.F = 1;
  r.x.assign(64, 0x3f80u); // BF16 1.0
  r.weights.assign(36, 0x00);
  for (int s = 0; s < 4; ++s) r.weights[s] = 0x38; // scale 1.0
  for (int j = 0; j < 32; ++j) r.weights[4 + j] = 0x31; // .5 and 1.5
  const auto dot = cpu_fp64_row(r, 0);
  if (dot.dot != 64. || dot.sumabs != 64.)
    error("FP64 dot/nibble mapping selftest failed");
  std::cout << "CPU_SELFTEST PASS RNE, UE4M3/E2M1, GGUF nibble orientation, FP64 dot\n";
}

} // namespace gemv_real
