// CPU-only policy witness for future K>1 greedy verification. This test does
// not link the MTP CUDA path or exercise a real PP2 DecodeState transaction.
// A generated correction/bonus token remains UNPROCESSED until the next
// target decode_step. A verified EOS stops without processing EOS itself.
#include "../mtp_gpu.hpp"
#include "../pp2_fwd.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(!mtp_gpu::target_batch_rollback_supported(),
              "K>1 target rollback must stay unavailable until PP2 implements it");
static_assert(pp2::DecodeState::MAX_CONTEXT == mtp_gpu::MAX_CONTEXT);
static_assert(std::extent_v<decltype(pp2::DecodeState::S)> == 64);
static_assert(std::extent_v<decltype(pp2::DecodeState::CS)> == 64);
static_assert(std::extent_v<decltype(pp2::DecodeState::Kc)> == 64);
static_assert(std::extent_v<decltype(pp2::DecodeState::Vc)> == 64);

namespace {
constexpr int kLayers = 64;
constexpr int kVocabulary = 997;
int assertions = 0;
void require(bool ok, const char* why) {
  ++assertions;
  if (!ok) { std::cerr << "FAIL " << why << '\n'; std::abort(); }
}

struct Layer {
  uint64_t gdn = 0;                 // Stand-in for a full recurrent GDN matrix
  std::array<int,3> conv{{0,0,0}}; // Stand-in for actual convolution tail
  std::vector<int> keys, values;    // Stand-in for full-attention tensor VALUES
  int absolute_pos = -1;
  bool operator==(const Layer& rhs) const {
    return gdn==rhs.gdn && conv==rhs.conv && keys==rhs.keys &&
           values==rhs.values && absolute_pos==rhs.absolute_pos;
  }
};

struct Target {
  std::array<Layer,kLayers> layers{};
  std::vector<int> processed;
  uint64_t request = 0, epoch = 0;
  int next_pos = 0;
  bool terminal = false;
  bool operator==(const Target& rhs) const {
    return layers==rhs.layers && processed==rhs.processed &&
           request==rhs.request && epoch==rhs.epoch &&
           next_pos==rhs.next_pos && terminal==rhs.terminal;
  }
};

// The independent policy oracle has deterministic, position-dependent outputs.
// It is a synthetic teacher, NOT a numerical oracle for Qwen weights.
int teacher_next(const Target& t) {
  const auto& l=t.layers[0];
  const uint64_t value=l.gdn + (uint64_t)(t.next_pos*113 + 17) +
      (uint64_t)(t.processed.empty()?0:t.processed.back()*19);
  return static_cast<int>(value % kVocabulary);
}
bool process_input(Target& t, int tok) {
  if (t.terminal || tok<0 || tok>=kVocabulary || t.next_pos>=128) return false;
  const int pos=t.next_pos++;
  t.processed.push_back(tok);
  for (int i=0;i<kLayers;++i) {
    auto& l=t.layers[(size_t)i];
    l.absolute_pos=pos;
    if (i%4==3) {
      l.keys.push_back(tok+29*i+pos);
      l.values.push_back(tok-13*i-pos);
    } else {
      l.gdn=(l.gdn*1315423911ULL)^static_cast<uint64_t>(tok*997+pos*37+i);
      l.conv={{l.conv[1],l.conv[2],tok+i}};
    }
  }
  return true;
}

enum class Failure { none, cancelled, deadline, device_error };
struct Reply {
  bool committed=false;
  int accepted=0;                // number of proposals verified before output
  std::vector<int> emitted;      // target-verified tokens, last NOT processed
  bool stopped=false;
};

// A future transaction must verify against tentative state and atomically
// install only the processed prefix. This reference function deliberately
// exposes no production API. `proposals` may contain malicious or bad tokens.
Reply verify_policy(Target& authoritative, const Target& checkpoint,
                    const std::vector<int>& proposals, int eos,
                    Failure failure=Failure::none) {
  if (failure!=Failure::none || authoritative.request!=checkpoint.request ||
      authoritative.epoch!=checkpoint.epoch || !(authoritative==checkpoint) ||
      authoritative.terminal || proposals.empty() || proposals.size()>4)
    return {};
  Target tentative=checkpoint;
  Reply reply;
  for (int candidate:proposals) {
    const int verified=teacher_next(tentative);
    if (verified==eos) {
      reply.emitted.push_back(verified);
      tentative.terminal=true;
      authoritative=std::move(tentative);
      reply.committed=true;
      reply.stopped=true;
      return reply;
    }
    if (verified!=candidate) {
      reply.emitted.push_back(verified); // correction pending for next step
      authoritative=std::move(tentative);
      reply.committed=true;
      return reply;
    }
    reply.emitted.push_back(candidate);
    ++reply.accepted;
    if (!process_input(tentative,candidate)) return {}; // fail closed
  }
  const int bonus=teacher_next(tentative);
  reply.emitted.push_back(bonus); // bonus pending for NEXT decode_step
  if (bonus==eos) { tentative.terminal=true; reply.stopped=true; }
  authoritative=std::move(tentative);
  reply.committed=true;
  return reply;
}

// Independent serial reference: every emitted token is teacher-verified; only
// consumed tokens are applied to recurrent/conv/KV layers. Last output pending.
std::pair<Target,std::vector<int>> serial_teacher(Target seed, int max_out, int eos) {
  std::vector<int> out;
  for (int i=0;i<max_out;++i) {
    const int tok=teacher_next(seed);
    out.push_back(tok);
    if (tok==eos) { seed.terminal=true; break; }
    if (i+1<max_out) require(process_input(seed,tok),"serial teacher process");
  }
  return {seed,out};
}

Target seeded(int seed, int request, int epoch) {
  Target t;
  t.request=(uint64_t)request;
  t.epoch=(uint64_t)epoch;
  for (int i=0;i<3;++i) require(process_input(t,(seed+37*i)%kVocabulary),"seed input");
  return t;
}
std::vector<int> propose(const Target& seed, int K, int mismatch_at=-1) {
  Target cursor=seed;
  std::vector<int> out;
  for (int i=0;i<K;++i) {
    int tok=teacher_next(cursor);
    if (i==mismatch_at) tok=(tok+1)%kVocabulary;
    out.push_back(tok);
    require(process_input(cursor,tok),"draft proposal mock process");
  }
  return out;
}

void test_accepted_prefix_and_correction() {
  for (int trial=0;trial<128;++trial) {
    Target original=seeded(41+trial,8000+trial,trial+1);
    for (int k=1;k<=4;++k) {
      for (int reject=-1;reject<k;++reject) {
        const auto draft=propose(original,k,reject);
        Target current=original;
        const Reply r=verify_policy(current,original,draft,-1);
        require(r.committed && !r.stopped,"valid bounded verify commits");
        require(r.accepted==(reject<0?k:reject),"exact accepted prefix");
        require(r.emitted.size()==(size_t)(r.accepted+1),"correction or bonus exists");
        const auto serial=serial_teacher(original,r.accepted+1,-1);
        require(current==serial.first,"all target layer values equal serial target");
        require(r.emitted==serial.second,"emission is teacher-only and ordered");
        require(current.processed.size()==original.processed.size()+(size_t)r.accepted,
                "correction or bonus is UNPROCESSED");
      }
    }
  }
}

void test_eos_and_untrusted_proposals() {
  for (int seed=1;seed<=64;++seed) {
    const Target original=seeded(seed,9000+seed,6);
    const auto perfect=propose(original,4);
    for (int stop=0;stop<=4;++stop) {
      Target before_bonus=original;
      for (int i=0;i<stop;++i)
        require(process_input(before_bonus,perfect[(size_t)i]),"EOS expected prefix");
      const int eos=teacher_next(before_bonus);
      // If EOS occurs earlier by coincidence, independently find first EOS.
      Target current=original;
      Reply r=verify_policy(current,original,perfect,eos);
      require(r.committed && r.stopped,"verified EOS terminates");
      const auto serial=serial_teacher(original,5,eos);
      require(current==serial.first && r.emitted==serial.second,
              "EOS only after authoritative target verification");
      require(!process_input(current,123),"terminal target state blocks later work");
    }
    // A draft hallucinating EOS is not permission to end the target request.
    Target current=original;
    const int teacher=teacher_next(original);
    const int fake_eos=(teacher+1)%kVocabulary;
    const auto forged=std::vector<int>{fake_eos,5,6};
    const Reply r=verify_policy(current,original,forged,fake_eos);
    require(r.committed && !r.stopped && r.accepted==0,"unverified EOS rejected");
    require(r.emitted.size()==1 && r.emitted[0]==teacher && current==original,
            "target correction pending, no speculative state mutation");
  }
}

void test_cancellation_failures_and_ownership() {
  for (int i=0;i<64;++i) {
    const Target original=seeded(i+11,21000+i,31);
    const auto draft=propose(original,4);
    for (Failure fault:{Failure::cancelled,Failure::deadline,Failure::device_error}) {
      Target authoritative=original;
      const Reply r=verify_policy(authoritative,original,draft,-1,fault);
      require(!r.committed && r.emitted.empty() && authoritative==original,
              "cancel/deadline/device fault must not emit or mutate target");
    }
    Target modified=original;
    require(process_input(modified,33),"external mutation simulated");
    require(!verify_policy(modified,original,draft,-1).committed,
            "stale checkpoint cannot overwrite committed target");
    Target wrong_epoch=original;
    ++wrong_epoch.epoch;
    require(!verify_policy(wrong_epoch,original,draft,-1).committed,
            "epoch mismatch forbids stale asynchronous completion");
    Target wrong_request=original;
    ++wrong_request.request;
    require(!verify_policy(wrong_request,original,draft,-1).committed,
            "cross-request completion forbidden");
    Target invalid=original;
    require(!verify_policy(invalid,original,{},-1).committed,
            "empty batch rejected");
    require(!verify_policy(invalid,original,{1,2,3,4,5},-1).committed,
            "oversized batch rejected");
  }
}

void test_mutants_are_detected() {
  const Target baseline=seeded(22,61001,1);
  const auto draft=propose(baseline,4,2);
  Target valid=baseline;
  const auto resolution=verify_policy(valid,baseline,draft,-1);
  require(resolution.accepted==2,"middle reject fixture");

  // Mutant 1: stage ALL four candidates in-place and truncate visible KV
  // length only; forgotten recurrent GDN/conv and KV contents must diverge.
  Target length_only=baseline;
  for (int token:draft) require(process_input(length_only,token),"bad staged input");
  length_only.processed.resize(baseline.processed.size()+2);
  length_only.next_pos=baseline.next_pos+2;
  for (int i=3;i<kLayers;i+=4) {
    auto& layer=length_only.layers[(size_t)i];
    layer.keys.resize(length_only.processed.size());
    layer.values.resize(length_only.processed.size());
  }
  require(!(length_only==valid),"negative mutant KV-length-only fails");
  require(length_only.layers[0].gdn!=valid.layers[0].gdn,
          "negative mutant GDN not restored");
  require(length_only.layers[0].conv!=valid.layers[0].conv,
          "negative mutant conv history not restored");
  // When rejected draft tokens are incorrectly retained, their KV VALUES are
  // also wrong even if a caller manually restores the visible cursor.
  const auto wrong_draft=propose(baseline,4,0);
  Target incorrect_prefix=baseline;
  require(process_input(incorrect_prefix,wrong_draft[0]),"bad prefix retained");
  incorrect_prefix.next_pos=baseline.next_pos; // mutant: host cursor-only rewind
  require(incorrect_prefix.layers[3].keys!=baseline.layers[3].keys,
          "negative mutant rejected KV values retained");

  // Mutant 2: process the emitted correction immediately (one-step drift).
  Target correction_processed=valid;
  require(process_input(correction_processed,resolution.emitted.back()),"bad eager correction");
  require(!(correction_processed==valid),"negative mutant eager correction detected");

  // Mutant 3: trust draft EOS rather than checking teacher. Target disagrees.
  const int fake=(teacher_next(baseline)+1)%kVocabulary;
  Target eos_target=baseline;
  const Reply r=verify_policy(eos_target,baseline,{fake},fake);
  require(r.committed && !r.stopped,"negative mutant unchecked EOS detected");
  require(eos_target==baseline,"rejection leaves layer state untouched");

  // Mutant 4: position-offset verification compares the wrong teacher.
  Target cursor=baseline;
  const int required=teacher_next(cursor);
  require(process_input(cursor,required),"advance one for off-by-one mutant");
  Target actual=baseline;
  const auto verified=verify_policy(actual,baseline,{required},-1);
  require(verified.committed && verified.accepted==1,
          "correct-position target accepts its own next token");
  require(teacher_next(cursor)!=required,
          "off-by-one target logit index changes the decision in fixture");
}

void test_interleaved_requests() {
  std::array<Target,8> states{};
  std::array<Target,8> originals{};
  std::array<std::vector<int>,8> proposals{};
  for (int i=0;i<8;++i) {
    states[(size_t)i]=seeded(100+i,10000+i,80+i);
    originals[(size_t)i]=states[(size_t)i];
    proposals[(size_t)i]=propose(states[(size_t)i],4,i%4);
  }
  for (int i=7;i>=0;--i) {
    const auto r=verify_policy(states[(size_t)i],originals[(size_t)i],
                               proposals[(size_t)i],-1);
    require(r.committed,"request commits independently");
    const auto serial=serial_teacher(originals[(size_t)i],r.accepted+1,-1);
    require(states[(size_t)i]==serial.first && r.emitted==serial.second,
            "interleaved requests remain isolated");
  }
}
} // namespace

int main() {
  test_accepted_prefix_and_correction();
  test_eos_and_untrusted_proposals();
  test_cancellation_failures_and_ownership();
  test_mutants_are_detected();
  test_interleaved_requests();
  std::cout << "PASS MTP_BATCHED_ATOMICITY_CPU_ONLY assertions=" << assertions
            << " target_K_gt_1=BLOCKED GPU=NOT_RUN throughput=NOT_MEASURED\n";
}
