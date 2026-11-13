#!/usr/bin/env python3
"""CPU/OFFLINE source-run harness tests; all urllib responses are injected.

Never opens sockets, launches HERMES, samples real CUDA, or contacts services.
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import re
import threading
import unittest
import urllib.error
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "scratch_http_soak.py"
spec = importlib.util.spec_from_file_location("scratch_http_soak", SOURCE)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


class MockClock:
    def __init__(self):
        self.value = 0.0
        self.lock = threading.Lock()

    def now(self):
        with self.lock:
            self.value += 0.01
            return self.value

    def sleep(self, amount):
        with self.lock:
            self.value += amount


class Response:
    def __init__(self, url, body, status=200, content_type="application/json",
                 redirect_url=None):
        self.body = body if isinstance(body, bytes) else (
            json.dumps(body).encode("utf-8") if content_type == "application/json"
            else str(body).encode("utf-8"))
        self.status = status
        self.url = redirect_url or url
        self.headers = {"Content-Type": content_type}

    def geturl(self):
        return self.url

    def read(self, maximum=-1):
        return self.body[:maximum] if maximum > -1 else self.body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeUrlOpener:
    """Thread-safe deterministic stand-in for urllib's opener.open()."""
    def __init__(self, *, clients=4, vram=None):
        self.clients = clients
        self.vram = vram or [(11000, 12000)]
        self.lock = threading.Lock()
        self.called = []
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.per_case = {}
        self.status = 200
        self.bad_content_type = False
        self.redirect = False
        self.bad_response = None
        self.bad_usage = False
        self.drift = False
        self.inject_error = False
        self.external_counter = False
        self.busy_at_start = False
        self.busy_after = False
        self.missing_metric = False
        self.sampled_after_each_round = []

    def _metrics(self):
        n = self.requests
        index = min(n // self.clients, len(self.vram) - 1)
        gpu0, gpu1 = self.vram[index]
        active = int(self.busy_at_start or (self.busy_after and n > 0))
        if not self.busy_at_start and n == 0:
            active = 0
        counters = {
            "hermes_server_ready": 1,
            "hermes_requests_total": n + int(self.external_counter and n > 0),
            "hermes_requests_accepted_total": n,
            "hermes_requests_completed_total": n,
            "hermes_requests_rejected_total": 0,
            "hermes_requests_cancelled_total": 0,
            "hermes_stream_backpressure_total": 0,
            "hermes_inference_errors_total": int(self.inject_error and n > 0),
            "hermes_prefill_failed_total": 0,
            "hermes_prompt_tokens_total": self.prompt_tokens,
            "hermes_completion_tokens_total": self.completion_tokens,
            "hermes_requests_active": active,
            "hermes_requests_queued": 0,
        }
        if self.missing_metric:
            del counters["hermes_inference_errors_total"]
        lines = [f"{key} {value}" for key, value in counters.items()]
        lines.extend((f'hermes_vram_used_mib{{gpu="0"}} {gpu0}',
                      f'hermes_vram_used_mib{{gpu="1"}} {gpu1}'))
        if n and n % self.clients == 0:
            self.sampled_after_each_round.append((gpu0, gpu1))
        return "\n".join(lines) + "\n"

    def open(self, req, timeout):
        del timeout
        path = req.full_url.removeprefix(module.BASE_URL)
        with self.lock:
            self.called.append((req.full_url, req.get_method()))
            if path == "/health":
                return Response(req.full_url, {"status": "ok", "ready": True})
            if path == "/v1/models":
                return Response(req.full_url, {"data": [{"id": "hermes-27"}]})
            if path == "/metrics":
                return Response(req.full_url, self._metrics(), content_type="text/plain")
            if path != "/v1/chat/completions":
                raise AssertionError(f"unapproved fake HTTP path {path!r}")
            if req.get_method() != "POST":
                raise AssertionError("chat request must POST")
            if self.status != 200:
                raise urllib.error.HTTPError(req.full_url, self.status, "injected HTTP failure",
                                             {}, io.BytesIO(b'{"error":"fail"}'))
            body = json.loads(req.data)
            if body["model"] != "hermes-27" or body["temperature"] != 0 or body["stream"] is not False:
                raise AssertionError("unexpected nondeterministic/streaming load request")
            prompt = body["messages"][0]["content"]
            match = re.search(r"N([48]) case-(\d+)", prompt)
            if match is None:
                raise AssertionError("prompt must include stable N and slot, no round")
            key = (int(match.group(1)), int(match.group(2)))
            seen = self.per_case.get(key, 0)
            self.per_case[key] = seen + 1
            self.requests += 1
            self.prompt_tokens += 33
            self.completion_tokens += 2
            output = "5" if self.drift and key == (4, 1) and seen >= 1 else "4"
            usage = {"prompt_tokens": 33, "completion_tokens": 2, "total_tokens": 35}
            if self.bad_usage:
                usage["total_tokens"] += 1
            payload = {
                "id": f"mock-{self.requests}",
                "object": "chat.completion",
                "model": "hermes-27",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": output},
                             "finish_reason": "stop"}],
                "usage": usage,
            }
            if self.bad_response is not None:
                payload = self.bad_response
            return Response(req.full_url, payload,
                            content_type="text/html" if self.bad_content_type else "application/json",
                            redirect_url="http://outside.example/" if self.redirect else None)


def config(*, rounds=3, clients=(4,), **kwargs):
    return module.Config(clients=clients, rounds=rounds, **kwargs)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.clock = MockClock()

    def run_mock(self, fake, *, cfg=None):
        return module.run_soak(cfg or config(), opener=fake,
                               now=self.clock.now, sleep=self.clock.sleep)

    def test_N4_N8_three_rounds_stable_hash_and_metrics(self):
        fake = FakeUrlOpener(clients=4)
        # For N8, fake request counters remain global and exact; memory stays flat.
        rows = self.run_mock(fake, cfg=config(rounds=3, clients=(4, 8)))
        self.assertEqual(len(rows), 6)
        self.assertEqual(fake.requests, 36)
        self.assertTrue(all(r["active"] == 0 and r["queued"] == 0 for r in rows))
        self.assertTrue(all(r["inference_errors_delta"] == 0 for r in rows))
        self.assertEqual([r["clients"] for r in rows], [4, 4, 4, 8, 8, 8])
        self.assertEqual([r["vram_used_mib"] for r in rows], [[11000, 12000]] * 6)
        self.assertTrue(all(r["requests"] == len(r["cases"]) for r in rows))
        self.assertEqual(len({r["cases"][0]["case_sha256"] for r in rows[:3]}), 1)
        self.assertEqual(len({r["cases"][0]["content_sha256"] for r in rows[3:]}), 1)
        self.assertTrue(all(r["completion_tokens"] == 2*r["clients"] for r in rows))
        self.assertGreaterEqual(len(fake.sampled_after_each_round), 6)
        self.assertTrue(all(url.startswith(module.BASE_URL + "/") for url, _ in fake.called))

    def test_fail_closed_only_exact_localhost_endpoint(self):
        fake = FakeUrlOpener()
        bad = [
            "http://localhost:18127", "http://127.0.0.1:18128", "https://127.0.0.1:18127",
            "http://127.0.0.2:18127", "http://127.0.0.1:18127/metrics",
            "http://127.0.0.1:18127@bad.invalid", "http://127.0.0.1:18127/?url=http://bad",
            "http://[::1]:18127", "http://127.0.0.1:18127#frag",
        ]
        for url in bad:
            with self.subTest(url=url), self.assertRaisesRegex(module.SoakFailure, "only exact"):
                self.run_mock(fake, cfg=config(base_url=url))
        self.assertEqual(fake.called, [])

    def test_bound_concurrency_rounds_and_tokens(self):
        for cfg in (config(clients=(1,)), config(clients=(4, 4)),
                    config(rounds=1), config(rounds=13),
                    config(max_tokens=0), config(max_tokens=65),
                    config(timeout=121), config(vram_allowance_mib=-1),
                    config(vram_warmup_limit_mib=1025),
                    config(drain_timeout=61), config(rising_steps=7)):
            with self.subTest(cfg=cfg), self.assertRaises(module.SoakFailure):
                cfg.checked()

    def test_preflight_rejects_active_server_before_posts(self):
        fake = FakeUrlOpener()
        fake.busy_at_start = True
        # A nonzero preflight gauge must reject, including before any POST.
        fake.busy_after = False
        # Fake metrics actively sets occupied on initial sample.
        fake._metrics = lambda: module_test_metrics(active=1)
        with self.assertRaisesRegex(module.SoakFailure, "already has active"):
            self.run_mock(fake)
        self.assertEqual(fake.requests, 0)

    def test_malformed_missing_duplicate_gpu_or_error_metrics(self):
        valid = module_test_metrics()
        for data in (
            valid.replace("hermes_inference_errors_total 0\n", ""),
            valid.replace('hermes_vram_used_mib{gpu="1"} 12000\n', ""),
            valid + 'hermes_vram_used_mib{gpu="0"} 100\n',
            valid.replace("hermes_requests_active 0", "hermes_requests_active NaN"),
            valid.replace("hermes_requests_queued 0", "hermes_requests_queued -1"),
        ):
            with self.subTest(data=data[-60:]), self.assertRaises(module.SoakFailure):
                module.parse_metrics(data.encode())

    def test_structural_chat_validation_strict(self):
        template = {
            "id": "abc", "object": "chat.completion", "model": "hermes-27",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "4"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 33, "completion_tokens": 2, "total_tokens": 35},
        }
        self.assertEqual(module.validate_chat(template, 8, "hermes-27")["completion_tokens"], 2)
        for name, mutate in (
            ("wrong model", lambda o: o.update(model="other")),
            ("cancelled", lambda o: o["choices"][0].update(finish_reason="cancelled")),
            ("empty", lambda o: o["choices"][0]["message"].update(content="  ")),
            ("missing usage", lambda o: o.pop("usage")),
            ("negative usage", lambda o: o["usage"].update(prompt_tokens=-1)),
            ("bool usage", lambda o: o["usage"].update(completion_tokens=True)),
            ("wrong total", lambda o: o["usage"].update(total_tokens=36)),
            ("too many tokens", lambda o: o["usage"].update(completion_tokens=9, total_tokens=42)),
        ):
            obj = json.loads(json.dumps(template))
            mutate(obj)
            with self.subTest(name=name), self.assertRaises(module.SoakFailure):
                module.validate_chat(obj, 8, "hermes-27")

    def test_deterministic_content_drift_rejected(self):
        fake = FakeUrlOpener()
        fake.drift = True
        with self.assertRaisesRegex(module.SoakFailure, "deterministic output or usage drift"):
            self.run_mock(fake)
        self.assertGreaterEqual(fake.requests, 8)

    def test_http_errors_content_type_redirect_and_usage(self):
        cases = [
            ("HTTP", lambda f: setattr(f, "status", 503), "HTTP 503"),
            ("type", lambda f: setattr(f, "bad_content_type", True), "Content-Type"),
            ("redirect", lambda f: setattr(f, "redirect", True), "redirect"),
            ("usage", lambda f: setattr(f, "bad_usage", True), "invalid total"),
        ]
        for name, mutate, expected in cases:
            fake = FakeUrlOpener()
            mutate(fake)
            with self.subTest(name=name), self.assertRaisesRegex(module.SoakFailure, expected):
                self.run_mock(fake)

    def test_inference_error_and_external_request_counter_rejected(self):
        for name, mutate in (
            ("errors", lambda f: setattr(f, "inject_error", True)),
            ("external", lambda f: setattr(f, "external_counter", True)),
        ):
            fake = FakeUrlOpener()
            mutate(fake)
            with self.subTest(name=name), self.assertRaisesRegex(module.SoakFailure, "expected delta"):
                self.run_mock(fake)

    def test_bounded_drain_timeout_after_round(self):
        fake = FakeUrlOpener()
        fake.busy_after = True
        with self.assertRaisesRegex(module.SoakFailure, "drain timeout"):
            self.run_mock(fake, cfg=config(drain_timeout=0.2, poll_interval=0.05))
        self.assertEqual(fake.requests, 4)

    def test_drain_requires_two_zero_gauges_and_stable_vram(self):
        history = [
            module_test_metrics(active=1),
            module_test_metrics(vram=(11040, 12050)),
            module_test_metrics(vram=(11000, 12000)),
            module_test_metrics(vram=(11000, 12000)),
        ]
        class FakeHttpMetrics:
            reads = 0
            def fetch(self, path):
                if path != "/metrics":
                    raise AssertionError("unexpected path")
                index = min(self.reads, len(history) - 1)
                self.reads += 1
                return (200, history[index].encode(), "text/plain")

        http = FakeHttpMetrics()
        answer = module.wait_for_drain(http, 2, .05, now=self.clock.now,
                                       sleep=self.clock.sleep)
        self.assertEqual(http.reads, 4)
        self.assertEqual(answer.vram, (11000, 12000))

    def test_urllib_default_disables_external_proxies_and_redirects(self):
        with patch.object(module.urllib.request, "build_opener") as builder:
            module.HttpClient(config())
        handlers = builder.call_args.args
        self.assertEqual(len(handlers), 2)
        self.assertIsInstance(handlers[0], module.urllib.request.ProxyHandler)
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsInstance(handlers[1], module.DenyRedirects)
        with self.assertRaisesRegex(module.SoakFailure, "redirect refused"):
            handlers[1].redirect_request(None, None, 302, "redirect", {},
                                         "http://outside.example/")

    def test_warmup_cache_growth_and_plateau_are_not_proof_of_no_leak(self):
        fake = FakeUrlOpener(vram=[
            (11000, 12000), (11030, 12080), (11030, 12080), (11030, 12080)])
        rows = self.run_mock(fake)
        self.assertEqual(rows[0]["vram_assessment"], "warmup_cache_unclassified")
        self.assertEqual(rows[1]["vram_assessment"], "cache_or_allocator_plateau_not_proven_leak_free")
        self.assertEqual([row["vram_used_mib"] for row in rows],
                         [[11030, 12080]] * 3)

    def test_new_N8_cohort_rewarms_larger_bounded_memory_plateau(self):
        # A new workload shape may have a different allocator high-water.
        # Do not mislabel a one-time N4->N8 jump as a persistent leak, but
        # still check N8 rounds 2+ for further monotone or absolute growth.
        samples = ([(11000, 12000)] * 5 + [(11100, 12100)] * 5)
        fake = FakeUrlOpener(clients=4, vram=samples)
        rows = self.run_mock(fake, cfg=config(rounds=3, clients=(4, 8)))
        self.assertEqual(rows[0]["vram_assessment"], "warmup_cache_unclassified")
        self.assertEqual(rows[3]["vram_assessment"], "warmup_cache_unclassified")
        self.assertEqual(rows[3]["vram_baseline_mib"], [11000, 12000])
        self.assertEqual(rows[3]["vram_post_warmup_reference_mib"], [11100, 12100])
        self.assertEqual(rows[4]["vram_assessment"],
                         "cache_or_allocator_plateau_not_proven_leak_free")

    def test_vram_absolute_postwarmup_growth_threshold(self):
        fake = FakeUrlOpener(vram=[
            (11000, 12000), (11040, 12000), (11040, 12066), (11040, 12066)])
        with self.assertRaisesRegex(module.SoakFailure, "GPU1 VRAM grew 66 MiB"):
            self.run_mock(fake)

    def test_unbounded_first_round_cache_growth_fails_closed(self):
        fake = FakeUrlOpener(vram=[
            (11000, 12000), (11300, 12000), (11300, 12000)])
        with self.assertRaisesRegex(module.SoakFailure, "GPU0 warmup VRAM rose >256 MiB"):
            self.run_mock(fake)
        self.assertEqual(fake.requests, 4)

    def test_vram_monotone_three_consecutive_rises(self):
        fake = FakeUrlOpener(vram=[
            (11000, 12000), (11000, 12000), (11010, 12000),
            (11020, 12000), (11030, 12000), (11030, 12000)])
        with self.assertRaisesRegex(module.SoakFailure, "GPU0 VRAM rose at least 8 MiB"):
            self.run_mock(fake, cfg=config(rounds=5))

    def test_checksums_depend_on_content_usage_and_finish(self):
        x = {"id": "1", "object": "chat.completion", "model": "hermes-27",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": "4"},
                          "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 33, "completion_tokens": 2, "total_tokens": 35}}
        a = module.validate_chat(x, 8, "hermes-27")
        x["usage"]["prompt_tokens"] += 1
        x["usage"]["total_tokens"] += 1
        b = module.validate_chat(x, 8, "hermes-27")
        self.assertEqual(a["content_sha256"], b["content_sha256"])
        self.assertNotEqual(a["case_sha256"], b["case_sha256"])
        x["choices"][0]["finish_reason"] = "length"
        c = module.validate_chat(x, 8, "hermes-27")
        self.assertNotEqual(c["case_sha256"], b["case_sha256"])

    def test_cli_wrong_host_fails_before_building_urlopener(self):
        with patch.object(module.urllib.request, "build_opener",
                          side_effect=AssertionError("would open a socket")):
            self.assertEqual(module.main(["--base-url", "http://0.0.0.0:18127"]), 1)


def module_test_metrics(*, active=0, vram=(11000, 12000)):
    entries = {k: 0 for k in module.REQUIRED_COUNTERS}
    entries.update({"hermes_server_ready": 1, "hermes_requests_active": active,
                    "hermes_requests_queued": 0})
    return ("\n".join(f"{key} {value}" for key, value in entries.items()) +
            f'\nhermes_vram_used_mib{{gpu="0"}} {vram[0]}\n' +
            f'hermes_vram_used_mib{{gpu="1"}} {vram[1]}\n')


if __name__ == "__main__":
    unittest.main()
