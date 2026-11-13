// CPU-only MTP metadata / rollback contract regression.
// This executable links NO CUDA libraries and runs NO inference. The actual
// batch verifier is intentionally unavailable and must stay fail-closed.
// Build: g++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -pthread
//        distributed/tests/test_mtp_rollback_contract.cpp -o /tmp/test_mtp_rollback
#include "../mtp_gpu.hpp"
#include "../pp2_fwd.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using mtp_gpu::MtpState;
using mtp_gpu::detail::can_append;
using mtp_gpu::detail::commit_append;
using mtp_gpu::detail::rewind_metadata;

static_assert(!std::is_copy_constructible_v<MtpState>,
              "MtpState owns device allocations and must not alias via copy");
static_assert(!std::is_copy_assignable_v<MtpState>,
              "MtpState must not double-free shared K/V via copy");
static_assert(!mtp_gpu::target_batch_rollback_supported(),
              "No real target batch verify+full-layer rollback API exists");
static_assert(pp2::DecodeState::MAX_CONTEXT == mtp_gpu::MAX_CONTEXT);
static_assert(std::extent_v<decltype(pp2::DecodeState::S)> == 64);
static_assert(std::extent_v<decltype(pp2::DecodeState::CS)> == 64);
static_assert(std::extent_v<decltype(pp2::DecodeState::Kc)> == 64);
static_assert(std::extent_v<decltype(pp2::DecodeState::Vc)> == 64);
static_assert(std::extent_v<decltype(pp2::DecodeState::klen)> == 64);

namespace {
int checks = 0;
void expect(bool condition, const char* description) {
  ++checks;
  if (!condition) {
    std::cerr << "FAIL " << description << "\n";
    std::abort();
  }
}

void init_dummy(MtpState& st, int capacity = 8, long origin = -1) {
  st.inited = true; // CPU-only metadata, NO GPU allocations
  st.capacity = capacity;
  st.len = 0;
  st.first_pos = origin;
  st.poisoned = false;
}

void test_rewind_zero_keeps_origin() {
  MtpState st;
  init_dummy(st, 4);
  expect(can_append(st, 11), "first append allowed");
  commit_append(st, 11);
  expect(can_append(st, 12), "second append contiguous");
  commit_append(st, 12);
  expect(st.len == 2 && st.first_pos == 11, "two entries with origin");
  expect(rewind_metadata(st, 0), "rewind to zero");
  expect(st.len == 0 && st.first_pos == 11,
         "rewind zero PRESERVES origin; reset is different operation");
  expect(!can_append(st, 13), "disallow skipped position after rewind zero");
  expect(!can_append(st, 0), "disallow moving origin after rewind zero");
  expect(can_append(st, 11), "replay original first position");
  commit_append(st, 11);
  expect(!can_append(st, 11), "reject repeated old position");
  expect(can_append(st, 12), "allow correct next position");
  expect(!rewind_metadata(st, 2), "rewind cannot grow history");
  expect(!rewind_metadata(st, -1), "negative rewind rejected");
  expect(rewind_metadata(st, 0), "can rewind repeated prefix");
}

void test_bounds_reset_poison_and_request_isolation() {
  MtpState a,b;
  init_dummy(a, 3); init_dummy(b, 2);
  expect(can_append(a, 1), "a first");
  commit_append(a, 1);
  commit_append(a, 2);
  commit_append(a, 3);
  expect(!can_append(a, 4), "at capacity");
  expect(can_append(b, 31), "other request independently initialized");
  commit_append(b, 31);
  expect(b.len == 1 && b.first_pos == 31, "b unaffected");
  expect(rewind_metadata(a, 1), "a rewinds");
  expect(b.len == 1 && b.first_pos == 31, "b still unaffected");
  expect(!can_append(a, -1), "negative abs position");
  expect(!can_append(a, (long)mtp_gpu::MAX_CONTEXT), "past context limit");
  expect(!can_append(a, 99), "wrong contiguous origin");
  a.poisoned = true; // simulates any failed CUDA call after a KV write
  expect(!can_append(a, 2), "poison rejects all further appends");
  expect(!rewind_metadata(a, 0), "poison cannot be cleared by rewind");
  expect(can_append(b, 32), "poison isolated per request");
  // CPU-only reset mirrors the production reset semantics.
  mtp_gpu::detail::reset_metadata(a);
  expect(a.poisoned && !can_append(a, 5), "reset cannot unpoison CUDA state");
  expect(!can_append(b, 33), "b cannot gap its own history");
}

void test_empty_invalid_state() {
  MtpState st;
  expect(!can_append(st, 1), "no allocation => reject");
  expect(!rewind_metadata(st, 0), "no allocation => reject rewind");
  init_dummy(st, 1);
  expect(can_append(st, 0), "position zero allowed");
  commit_append(st, 0);
  expect(!can_append(st, 1), "capacity one bounded");
  expect(!can_append(st, 0), "no repeat at capacity");
  MtpState corrupted;
  init_dummy(corrupted,2,mtp_gpu::MAX_CONTEXT);
  expect(!can_append(corrupted,0),"corrupted origin outside max context");
  expect(!rewind_metadata(corrupted,0),"cannot rewind corrupt first position");
  corrupted.first_pos=-1;corrupted.len=1;
  expect(!can_append(corrupted,0),"nonzero history without origin");
  expect(!rewind_metadata(corrupted,0),"no rewind missing origin");
  corrupted.first_pos=mtp_gpu::MAX_CONTEXT-1;
  corrupted.len=2;
  expect(!rewind_metadata(corrupted,0),"reject last_pos beyond context");
}

void test_host_state_concurrent_metadata() {
  MtpState st;
  init_dummy(st, 512, 40);
  const int workers=8, increments=32;
  std::vector<std::thread> threads;
  for(int t=0;t<workers;++t) threads.emplace_back([&]{
    for(int i=0;i<increments;++i) {
      std::lock_guard<std::mutex> lk(st.mutex);
      const long next=st.first_pos+st.len;
      expect(can_append(st,next), "guarded next");
      commit_append(st,next);
    }
  });
  for(auto& t:threads)t.join();
  expect(st.len==workers*increments && st.first_pos==40,
         "threaded metadata no lost updates");
}

// Independent, deliberately tiny target mock: every layer is mutated to prove
// that merely decrementing a visible K length after rejection is not rollback.
constexpr int LAYERS = 64;
struct MockLayer {
  int pos=-1;
  uint64_t s=0; // GDN matrix-state digest proxy (not actual 128x128)
  std::array<int,3> cs{{0,0,0}};
  std::vector<int> K,V; // full attention KV histories
  bool operator==(const MockLayer&) const = default;
};
struct MockTarget {
  std::array<MockLayer,LAYERS> layers{};
  std::vector<int> committed;
  int next_pos=0;
  bool terminal=false;
  uint64_t request_scope=0;
  bool operator==(const MockTarget&) const = default;

  bool append(int tok, uint64_t scope) {
    if(terminal || scope!=request_scope || tok<0 || committed.size()>=16)
      return false;
    const int pos=next_pos++;
    committed.push_back(tok);
    for(int i=0;i<LAYERS;++i) {
      auto& l=layers[i];
      l.pos=pos;
      if(i%4==3){
        l.K.push_back(tok + i*31 + pos);
        l.V.push_back(tok - i*19 - pos);
      } else {
        l.cs={{l.cs[1],l.cs[2],tok+i}};
        l.s=(l.s*1315423911ULL) ^ static_cast<uint64_t>(tok*997+pos*31+i);
      }
    }
    return true;
  }
};

// In a hypothetical real batch implementation, target speculation must be
// transactional over all fields, not only klen. This policy model simulates
// the REQUIRED behavior, not a callable PP2 function.
struct Transaction {
  MockTarget original, candidate;
  int pending=0;
  bool failed=false;
  explicit Transaction(const MockTarget& target):
    original(target),candidate(target){}
  bool stage(int tok) {
    if(failed || pending>=4) {failed=true; return false;}
    if(!candidate.append(tok,candidate.request_scope)){failed=true;return false;}
    ++pending; return true;
  }
  bool resolve(MockTarget& authoritative, int accepted, int replacement,
               bool eos=false, bool cancel=false) {
    if(failed || accepted<0 || accepted>pending
       || authoritative.request_scope != original.request_scope
       || !(authoritative==original)) return false;
    // Atomic full-state rollback: restart from known committed snapshot and
    // replay only verified accepted prefix, optionally target correction.
    if(cancel) return true; // no mutation of authoritative state
    MockTarget corrected=original;
    for(int i=0;i<accepted;++i) {
      const int tok=candidate.committed[original.committed.size()+i];
      if(!corrected.append(tok,corrected.request_scope))return false;
    }
    if(accepted<pending) {
      if(replacement<0 || !corrected.append(replacement,corrected.request_scope))
        return false;
    }
    corrected.terminal=eos;
    authoritative=std::move(corrected);
    return true;
  }
};
void test_all_layer_rollback_requirement() {
  MockTarget baseline;
  baseline.request_scope=7001;
  expect(baseline.append(11,7001), "baseline seed");
  MockTarget correct=baseline;
  Transaction tx(baseline);
  expect(tx.stage(20) && tx.stage(21) && tx.stage(22), "three staged");
  expect(tx.resolve(correct,1,99), "accept one + correct second");
  MockTarget replay=baseline;
  expect(replay.append(20,7001) && replay.append(99,7001),
         "serial authoritative replay");
  expect(correct==replay,"transactional entire layer state matches serial");
  // Deliberately broken strategy: mutate target state in place on all
  // speculative candidates and rewind only full-KV cache host lengths.
  MockTarget broken=tx.candidate;
  for(int i=0;i<LAYERS;++i) if(i%4==3) {
    broken.layers[i].K.resize(2);
    broken.layers[i].V.resize(2);
  }
  broken.next_pos=2;
  broken.committed.resize(2);
  expect(!(broken==replay),
         "rewind only K length does NOT restore GDN S and conv histories");
  expect(broken.layers[0].s!=replay.layers[0].s,
         "concrete surviving poisoned GDN state");
  expect(broken.layers[0].cs!=replay.layers[0].cs,
         "concrete surviving poisoned convolution tail");
  expect(broken.layers[3].K==replay.layers[3].K ? false : true,
         "replacing a rejected token must fix K VALUES not just K length");
}
void test_eos_cancel_fence_rejection_matrix() {
  const std::vector<int> seed={12,13};
  const std::array<std::array<int,4>,5> proposed{{
    {{4,5,6,7}},{{9,8,7,6}},{{10,11,12,13}},{{5,6,7,8}}
    ,{{1,3,5,7}}
  }};
  for(int scenario=0;scenario<5;++scenario){
    MockTarget target;
    target.request_scope=static_cast<uint64_t>(900+scenario);
    for(int tok:seed)expect(target.append(tok,target.request_scope),"seed");
    const MockTarget before=target;
    Transaction tx(target);
    for(int token:proposed[scenario])expect(tx.stage(token),"stage bounded");
    const int accept=scenario; // 0 rejected, partial prefixes, 4 all accepted
    const int replace=(accept<4?100+scenario:-1);
    expect(tx.resolve(target,accept,replace,scenario==2 || scenario==4),
           "resolve multiple accepted counts");
    MockTarget expected=before;
    for(int j=0;j<accept;++j)
      expect(expected.append(proposed[scenario][j],expected.request_scope),"accepted");
    if(accept<4)expect(expected.append(replace,expected.request_scope),"corrected");
    expected.terminal=(scenario==2 || scenario==4);
    expect(expected==target,"exact reference target state after correction");
    if(scenario==2 || scenario==4)
      expect(!target.append(42,target.request_scope),"no decoding after eos");
    Transaction cancel_test(before);
    expect(cancel_test.stage(77),"cancel candidate");
    MockTarget unchanged=before;
    expect(cancel_test.resolve(unchanged,0,99,false,true) && unchanged==before,
           "cancel leaves every layer unchanged");
    MockTarget foreign=before;
    ++foreign.request_scope;
    expect(!cancel_test.resolve(foreign,0,99),
           "stale request scope rejected");
    MockTarget externally_modified=before;
    expect(externally_modified.append(12,externally_modified.request_scope),"extra write");
    expect(!cancel_test.resolve(externally_modified,0,99),
           "stale transaction snapshot rejected");
    expect(!tx.resolve(target,-1,99),"negative accept rejected");
    expect(!tx.resolve(target,5,99),"overaccept rejected");
  }
}
void test_deterministic_full_prefix_matrix() {
  constexpr int KMAX=4, ROUNDS=256;
  std::mt19937 rng(0x4D54504Fu);
  for(int trial=0;trial<ROUNDS;++trial){
    MockTarget initial;
    initial.request_scope=static_cast<uint64_t>(3400+trial);
    expect(initial.append(10+trial%100,initial.request_scope),"matrix seed");
    Transaction tx(initial);
    const int k=1+static_cast<int>(rng()%KMAX);
    std::vector<int> candidates;
    for(int j=0;j<k;++j) {
      const int tok=5000+trial*5+j;
      candidates.push_back(tok);
      expect(tx.stage(tok),"matrix bounded candidate");
    }
    const int accept=static_cast<int>(rng()%(k+1));
    const bool eos=(trial%11==0);
    const bool cancel=(trial%17==0);
    const int correction=30000+trial;
    MockTarget actual=initial;
    expect(tx.resolve(actual,accept,correction,eos,cancel),
           "matrix commit or cancellation");
    MockTarget expected=initial;
    if(!cancel){
      for(int j=0;j<accept;++j)
        expect(expected.append(candidates[j],expected.request_scope),
               "matrix accepted target");
      if(accept<k)
        expect(expected.append(correction,expected.request_scope),
               "matrix correction");
      expected.terminal=eos;
    }
    expect(actual==expected,"matrix all layers and cursor match serial replay");
  }
}
void test_interleaved_requests() {
  constexpr int REQUESTS=8;
  std::array<MockTarget,REQUESTS> targets{};
  for(int r=0;r<REQUESTS;++r)
    targets[r].request_scope=static_cast<uint64_t>(r+2000);
  std::vector<std::thread> jobs;
  for(int r=0;r<REQUESTS;++r)jobs.emplace_back([&,r]{
    for(int t=0;t<3;++t) {
      expect(targets[r].append(10*r+t,targets[r].request_scope),"request seed");
    }
    Transaction tx(targets[r]);
    expect(tx.stage(400+r) && tx.stage(500+r),"multiuser candidate");
    const int accept=r%2;
    expect(tx.resolve(targets[r],accept,600+r),"multiuser resolved");
  });
  for(auto& j:jobs)j.join();
  for(int r=0;r<REQUESTS;++r) {
    MockTarget ref;ref.request_scope=static_cast<uint64_t>(r+2000);
    for(int t=0;t<3;++t)expect(ref.append(10*r+t,ref.request_scope),"reference seed");
    if(r%2)expect(ref.append(400+r,ref.request_scope),"accepted");
    expect(ref.append(600+r,ref.request_scope),"teacher correct");
    expect(targets[r]==ref,"multiuser no cross-request state contamination");
  }
}
} // namespace

int main(){
  // No GPU calls, no MTP or PP2 link, no speculative throughput assertions.
  test_rewind_zero_keeps_origin();
  test_bounds_reset_poison_and_request_isolation();
  test_empty_invalid_state();
  test_host_state_concurrent_metadata();
  test_all_layer_rollback_requirement();
  test_eos_cancel_fence_rejection_matrix();
  test_deterministic_full_prefix_matrix();
  test_interleaved_requests();
  std::cout << "PASS CPU_ONLY_MTP_MOCK_ROLLBACK checks=" << checks
            << " target_batch=BLOCKED no_gpu=true\n";
}
