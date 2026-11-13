# Manager lease v1 — offline patch gate, 2026-11-13

## Verdict

**PATCH / SOURCE-REVIEW GATE: PASS. PRODUCTION CUTOVER: NO-GO.**
This gate concerns only the static reviewable patch and offline mock tests.
The installed GPU Manager, Tabby, systemd services and GPUs were not changed.
No HERMES inference or service mutation was invoked.

The observed read-only source is /opt/hermes/gpu-manager/manager.py, SHA256:

15744e3b769c0f6cd5ed26c2b85e3f282871f4a24681abd324c8cbc329a73eef

A scratch source whose line endings changed failed the pinned SHA256 check,
as intended. The original was fetched read-only into a local scratch copy,
whose hash matched exactly; the scratch copy was never installed back.

## Whitespace repair and exact application

The old proposal (7,100 bytes) contained 17 single-space context lines and
a blank context line at EOF. Tracking the patch file itself therefore failed
Git's cached-diff whitespace check, even though the added Python code was
otherwise clean. The revised generator represents empty diff-context lines
in the Git-accepted no-space form, trims trailing context and decrements
both counts in the final hunk. It does not rewrite Python code.

| Offline gate | Outcome |
| --- | --- |
| Input source SHA256 matches pinned real manager.py | PASS |
| Regenerated patch has zero trailing whitespace lines | PASS |
| Regenerated patch has no blank line at EOF | PASS |
| Git apply --check on unmodified audited source | PASS |
| Git apply to an isolated copied manager.py | PASS |
| Applied copy equals staged patched source byte for byte | PASS |
| Git diff --cached --check with patch staged as a file | PASS |
| Python syntax compilation, Bash syntax check | PASS |
| Mock unit/integration suite | 115/115 PASS, local offline Python 3.13, 48.897 s |

The independently generated and scratch-applied patched manager.py SHA256:

2b78d53d159b06a472edfcb4227f8e2ab81033ecec4b363a1a74ff1652768357

The regenerated **ops/manager_lease_v1_proposal.patch** is 7,083 bytes,
SHA256 d7161ec7a91dd3d512f51a8b96218aea459ec1e22b80ace6e6d405b6d250d95e.

## Ready-to-commit owned-file manifest

| File | Change |
| --- | --- |
| ops/manager_lease_patch.py | Normalize blank context and adjust final hunk counts |
| ops/manager_lease_v1_proposal.patch | Whitespace-clean patch applicable to pinned source |
| ops/tests/test_manager_lease_patch.py | Git dry-run/apply and Git staged-whitespace regression tests |
| ops/manager_lease_runtime.py | Operator credential file must be private (0600 or stricter) |
| ops/tests/test_manager_lease_runtime.py | Refuse group-readable operator-token files |
| ops/MANAGER_LEASE_PATCH_GATE.md | This gate and security review |
| ops/manager_lease_REVIEW.md | Updated validation evidence |
| reports/PRODUCTION_DEPLOYMENT.md | Updated source-gate result |

Temporary staging folders and copied read-only source snapshots in
ops/.manager_lease_* are NOT deliverables and must not be committed.

SHA256 values of the primary code/patch files:

| File | SHA256 |
| --- | --- |
| ops/manager_lease_v1_proposal.patch | d7161ec7a91dd3d512f51a8b96218aea459ec1e22b80ace6e6d405b6d250d95e |
| ops/manager_lease_patch.py | 659d5ecebd4e9969428fb82111646845e2f912eb938eb99910f12e7018180118 |
| ops/manager_lease_runtime.py | e688d0b811725fd433702c77d37c4e8dd9a4401320adb56ca76500c3ff00b670 |
| ops/tests/test_manager_lease_patch.py | cf83dce6dfb9b027753e4b02d363c5470997e54a50ff5b99c76f9ce1b64011af |
| ops/tests/test_manager_lease_runtime.py | cfef276a543ca840f8d504e36fcc92b7ce0a8ab49f0333ffe33b80f9c9a27916 |

The prime separately verified the preceding 113-test ops version using
the server virtual environment. The follow-up modifications here were
verified in offline Windows Python only, not installed on the server.

## Regression audit: requirements still blocking production

**Ingress checker absent.** A local, independently trusted ingress source
must attest, within 10 seconds, all-tabby-clients scope, blocked=true,
inflight=0 for direct localhost, gateway, Tailscale and streaming requests.
Unknown direct client paths or a false/incomplete checker must deny lease
admission, renewal and recovery. A client-submitted proof or mock does not
count. The current prototype verifies the shape of the proof, not its
physical coverage.

**GPU/PID owner checker absent.** Before manager-initiated Tabby restart,
an independent privileged checker must enumerate BOTH cards, every CUDA
process and its verified PID/start-time identity, all HERMES children, and
any unknown owners. It must prove no HERMES CUDA owner and truly free GPUs.
After Tabby restore it must prove exclusive Tabby ownership without unknown
or still-running HERMES contexts. A process-name grep or GPU utilization
percentage cannot establish this. Any inaccessible, ambiguous or stale
evidence must preserve the fence.

**Persistence and trust gap.** All parent-directory write permissions,
checker binary ownership, invocation privileges, lease token permissions
and one-process manager semantics require review. Deletion of the durable
lease journal currently looks like a fresh install; an independently stored
fence and boot-time GPU-ownership gate are required before daily deployment.

**Manager-external restart gap.** The in-process patch cannot stop an
independent systemd Restart policy, another daemon with systemctl access,
direct Tabby clients or separate Character ID/gaming process launchers.
These paths must be audited in an explicitly approved maintenance window.

**Network/auth boundary.** The prototype introduces no new listening port;
it uses the existing privileged manager Bearer, a separate operator token
and strict loopback peer checks. The installed manager's existing listener
and firewall/Tailscale exposure still require security verification.

The result is **PATCH-READY FOR COMMIT / NOT SAFE FOR LIVE DEPLOYMENT**.
