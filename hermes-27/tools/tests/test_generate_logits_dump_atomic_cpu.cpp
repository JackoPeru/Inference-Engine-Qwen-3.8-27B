// CPU-only synthetic logits dump safety. Never uploads a GGUF or opens CUDA.
// Build as a separate TU with -ffunction-sections -Wl,--gc-sections:
// the renamed GPU CLI entrypoint is discarded, but dump_logits is exercised.
#define main hermes_model_cli_entry_not_called
#include "../hermes_generate.cpp"
#undef main

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
int checks = 0;
void require(bool ok, const char* message) {
  ++checks;
  if(!ok) throw std::runtime_error(message);
}

std::string read_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  require((bool)in, "cannot read test fixture");
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

void put_file(const std::filesystem::path& p, const std::string& v) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  require((bool)out, "cannot create test fixture");
  out.write(v.data(), static_cast<std::streamsize>(v.size()));
  require((bool)out, "cannot write test fixture");
}
} // namespace

int main() {
  char name[] = "/tmp/h27-logit-dump-private-XXXXXX";
  char* fresh = mkdtemp(name);
  if(!fresh) {
    std::cerr << "FAIL could not create private CPU scratch\n";
    return 2;
  }
  const auto root = std::filesystem::path(fresh);
  try {
    std::string error;
    std::vector<float> dummy(kVocab, 0.125f);
    const auto final0 = root / "prompt_0_step_0.f32";
    const auto partial0 = root / "prompt_0_step_0.f32.partial";
    require(dump_logits(root.string(), 0, 0, dummy, error),
            "first unique synthetic dump must succeed");
    const auto first = read_file(final0);
    require(first.size() == dummy.size() * sizeof(float),
            "first dump must have complete vector");
    const auto mode = std::filesystem::status(final0).permissions();
    require((mode & (std::filesystem::perms::group_all |
                     std::filesystem::perms::others_all)) ==
                 std::filesystem::perms::none,
            "private logits file must not expose data to group/other");
    require(!std::filesystem::exists(partial0), "no partial after success");

    dummy[0] = 0.75f;
    error.clear();
    require(!dump_logits(root.string(), 0, 0, dummy, error),
            "duplicate final must fail instead of silently replacing evidence");
    require(!error.empty(), "duplicate final failure reason required");
    require(read_file(final0) == first,
            "duplicate final must preserve exact originally written bytes");

    const auto partial1 = root / "prompt_0_step_1.f32.partial";
    const auto final1 = root / "prompt_0_step_1.f32";
    put_file(partial1, "KEEP_EXISTING_PARTIAL");
    error.clear();
    require(!dump_logits(root.string(), 0, 1, dummy, error),
            "preexisting partial must be preserved without truncation");
    require(read_file(partial1) == "KEEP_EXISTING_PARTIAL",
            "preexisting partial content changed");
    require(!std::filesystem::exists(final1),
            "final must not appear after exclusive partial open failure");

    const auto victim = root / "victim.txt";
    const auto final2 = root / "prompt_0_step_2.f32";
    put_file(victim, "KEEP_VICTIM");
    std::filesystem::create_symlink(victim.filename(), final2);
    error.clear();
    require(!dump_logits(root.string(), 0, 2, dummy, error),
            "preexisting symlink final must not be replaced");
    require(std::filesystem::is_symlink(final2),
            "preexisting symlink identity lost");
    require(read_file(victim) == "KEEP_VICTIM",
            "symlink target bytes changed");

    std::cout << "PASS synthetic exclusive logits dump checks=" << checks << "\n";
    std::filesystem::remove_all(root);
    return 0;
  } catch(const std::exception& ex) {
    std::cerr << "FAIL exclusive logits dump: " << ex.what() << "\n";
    std::filesystem::remove_all(root);
    return 1;
  }
}
