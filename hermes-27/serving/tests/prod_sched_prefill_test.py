#!/usr/bin/env python3
"""HTTP CPU oracle for --batched-prefill, both scheduler implementations.

No CUDA: uses the real worker-6 scheduling source against a deterministic
PP2 fake that records open, prefill, step, and close operations.
"""
import concurrent.futures
import json
import os
import subprocess
import unittest
import urllib.request

from prod_sched_state_test import (
    ARGS, OTHER, events, payload, post, running_server, trace_lines,
)


class PrefillServingCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file(ARGS.tokenizer)
        cls.output_id = cls.tokenizer.encode("hello", add_special_tokens=False).ids[0]
        cls.output_text = cls.tokenizer.decode([cls.output_id])

    def prompt_for_length(self, length):
        # The mock Jinja template outputs message.content without wrappers.
        # Qwen3.8 tokenizer encodes each " x" as exactly one token.
        prompt = " x" * length
        ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        self.assertEqual(len(ids), length, f"bad test fixture at T={length}")
        return prompt

    def message(self, prompt, max_tokens=4, **kwargs):
        body = payload(max_tokens, **kwargs)
        body["messages"][0]["content"] = prompt
        return body

    def assert_one_state(self, lines, *, prompt_length, output_tokens,
                         seeded=True, count=1):
        self.assertEqual(sum(line.startswith("open ") for line in lines), count, lines)
        self.assertEqual(sum(line == "close" for line in lines), count, lines)
        prefill = [line for line in lines if line.startswith("prefill ")]
        self.assertEqual(prefill, [f"prefill {prompt_length}"] * (count if seeded else 0), lines)
        steps = [int(line.split()[1]) for line in lines if line.startswith("step ")]
        first = prompt_length if seeded else 0
        expected = list(range(first, prompt_length + output_tokens - 1)) * count
        self.assertEqual(steps, expected, lines)
        self.assertEqual(
            len(lines), count * 2 + len(prefill) + len(steps), lines
        )

    def assert_mode(self, length, *, continuous, batched=True):
        flags = ("--batched-prefill",) if batched else ()
        with running_server(
            ids=[self.output_id], flags=flags, continuous=continuous,
            env_extra={"FAKE_STEP_MS": "0"},
        ) as (url, _, trace):
            prompt = self.prompt_for_length(length)
            body = self.message(prompt, 4)
            status, raw = post(url, body)
            self.assertEqual(status, 200, raw)
            response = json.loads(raw)
            self.assertEqual(response["choices"][0]["finish_reason"], "length")
            self.assertEqual(response["choices"][0]["message"]["content"],
                             self.tokenizer.decode([self.output_id] * 4))
            self.assertEqual(response["usage"]["prompt_tokens"], length)
            self.assertEqual(response["usage"]["completion_tokens"], 4)

            body = self.message(prompt, 4, stream=True, include_usage=True)
            status, raw = post(url, body)
            self.assertEqual(status, 200, raw)
            text, reasons, usages, done = events(raw)
            self.assertEqual(text, response["choices"][0]["message"]["content"])
            self.assertEqual(reasons, ["length"])
            self.assertEqual(done, 1)
            self.assertEqual(len(usages), 1)
            self.assertEqual(usages[0], response["usage"])
            self.assert_one_state(
                trace_lines(trace), prompt_length=length, output_tokens=4,
                seeded=batched and length <= 64, count=2,
            )
            with urllib.request.urlopen(url + "/metrics", timeout=10) as response:
                metrics = response.read().decode("utf-8")
            def metric(name):
                matching = [line for line in metrics.splitlines()
                            if line.startswith(name + " ")]
                self.assertEqual(len(matching), 1, metrics)
                return int(matching[0].split()[1])
            expected = 2 if batched and length <= 64 else 0
            self.assertEqual(metric("hermes_prefill_attempts_total"), expected)
            self.assertEqual(metric("hermes_prefill_seeded_total"), expected)
            self.assertEqual(metric("hermes_prefill_failed_total"), 0)
            self.assertEqual(metric("hermes_prefill_prompt_tokens_total"),
                             expected * length)

    def test_legacy_seeded_T1_T11_T64(self):
        for length in (1, 11, 64):
            with self.subTest(length=length):
                self.assert_mode(length, continuous=False)

    def test_continuous_seeded_T1_T11_T64(self):
        for length in (1, 11, 64):
            with self.subTest(length=length):
                self.assert_mode(length, continuous=True)

    def test_long_T65_falls_back_no_seed(self):
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                self.assert_mode(65, continuous=continuous)

    def test_default_no_flag_keeps_stepwise_prompt(self):
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                self.assert_mode(11, continuous=continuous, batched=False)

    def test_first_seeded_token_eos_finishes_once(self):
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                with running_server(
                    ids=[248046], flags=("--batched-prefill",), continuous=continuous,
                    env_extra={"FAKE_STEP_MS": "0"},
                ) as (url, _, trace):
                    body = self.message(self.prompt_for_length(11), 5,
                                        stream=True, include_usage=True)
                    status, raw = post(url, body)
                    self.assertEqual(status, 200, raw)
                    text, reasons, usage, done = events(raw)
                    self.assertEqual(text, "")
                    self.assertEqual(reasons, ["stop"])
                    self.assertEqual(len(usage), 1)
                    self.assertEqual(usage[0]["completion_tokens"], 1)
                    self.assertEqual(done, 1)
                    self.assert_one_state(
                        trace_lines(trace), prompt_length=11, output_tokens=1,
                    )

    def test_first_seeded_token_text_stop_and_utf8_parity(self):
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                with running_server(
                    ids=[self.output_id], flags=("--batched-prefill",),
                    continuous=continuous, env_extra={"FAKE_STEP_MS": "0"},
                ) as (url, _, trace):
                    body = self.message(
                        self.prompt_for_length(11), 5, stream=True,
                        stops=self.output_text, include_usage=True,
                    )
                    status, raw = post(url, body)
                    self.assertEqual(status, 200, raw)
                    text, reasons, usage, done = events(raw)
                    self.assertEqual(text, "")
                    self.assertEqual(reasons, ["stop"])
                    self.assertEqual(usage[0]["completion_tokens"], 1)
                    self.assertEqual(done, 1)
                    self.assert_one_state(trace_lines(trace), prompt_length=11,
                                          output_tokens=1)

    def test_prefill_failure_never_replays_poisoned_state(self):
        for continuous in (False, True):
            for failure in ("FAKE_PREFILL_FAIL", "FAKE_PREFILL_NONFINITE"):
                with self.subTest(continuous=continuous, failure=failure):
                    with running_server(
                        ids=[self.output_id], flags=("--batched-prefill",),
                        continuous=continuous, env_extra={failure: "1"},
                    ) as (url, _, trace):
                        status, raw = post(url, self.message(
                            self.prompt_for_length(11), stream=True))
                        self.assertIn(status, (500, 503), raw)
                        self.assertIn("error", json.loads(raw))
                        lines = trace_lines(trace)
                        self.assertEqual(sum(line.startswith("open ") for line in lines), 1, lines)
                        self.assertEqual(sum(line == "close" for line in lines), 1, lines)
                        self.assertEqual(sum(line == "prefill 11" for line in lines), 1, lines)
                        self.assertFalse(any(line.startswith("step ") for line in lines), lines)

    def test_post_seed_cancel_closes_state_once(self):
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                with running_server(
                    ids=[self.output_id], flags=("--batched-prefill",),
                    continuous=continuous, env_extra={"FAKE_STEP_MS": "30"},
                ) as (url, _, trace):
                    body = self.message(self.prompt_for_length(11), 60,
                                        stream=True, include_usage=True)
                    req = urllib.request.Request(
                        url + "/v1/chat/completions",
                        data=json.dumps(body).encode("utf-8"),
                        headers={"Content-Type": "application/json"}, method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=30) as response:
                        first = response.readline()
                        self.assertTrue(first.startswith(b"data: "), first)
                        ident = json.loads(first[6:])["id"]
                        status, raw = post(url, {"id": ident}, endpoint="/v1/cancel")
                        self.assertEqual(status, 202, raw)
                        text, reasons, usage, done = events(first + response.read())
                        self.assertEqual(reasons, ["cancelled"])
                        self.assertEqual(len(usage), 1)
                        self.assertEqual(done, 1)
                    lines = trace_lines(trace)
                    self.assertEqual(sum(line.startswith("open ") for line in lines), 1, lines)
                    self.assertEqual(sum(line == "close" for line in lines), 1, lines)
                    self.assertEqual(sum(line == "prefill 11" for line in lines), 1, lines)
                    self.assertTrue(all(int(line.split()[1]) >= 11
                                        for line in lines if line.startswith("step ")), lines)

    def test_multiuser_mixed_prompt_sizes_no_replay(self):
        # N6 mixed-context pressure (active N4). Each request has a separate
        # PP2 state, even while a long T65 request takes the ordinary path.
        requests = [(1, 3), (11, 4), (64, 2), (65, 3), (11, 2), (1, 4)]
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                with running_server(
                    ids=[self.output_id], flags=("--batched-prefill", "--active", "4",
                                                 "--queue", "16"),
                    continuous=continuous, env_extra={"FAKE_STEP_MS": "1"},
                ) as (url, _, trace):
                    def one(case):
                        length, maximum = case
                        body = self.message(self.prompt_for_length(length), maximum,
                                            stream=(length % 2 == 1),
                                            include_usage=(length % 2 == 1))
                        status, raw = post(url, body)
                        self.assertEqual(status, 200, raw)
                        if body["stream"]:
                            content, reasons, usages, done = events(raw)
                            self.assertEqual(done, 1)
                            self.assertEqual(reasons, ["length"])
                            self.assertEqual(len(usages), 1)
                            self.assertEqual(usages[0]["prompt_tokens"], length)
                            self.assertEqual(usages[0]["completion_tokens"], maximum)
                        else:
                            response = json.loads(raw)
                            content = response["choices"][0]["message"]["content"]
                            self.assertEqual(response["usage"]["prompt_tokens"], length)
                            self.assertEqual(response["usage"]["completion_tokens"], maximum)
                        self.assertEqual(content,
                                         self.tokenizer.decode([self.output_id] * maximum))

                    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                        list(pool.map(one, requests))
                    lines = trace_lines(trace)
                    self.assertEqual(sum(x.startswith("open ") for x in lines), 6, lines)
                    self.assertEqual(sum(x == "close" for x in lines), 6, lines)
                    seeded = sorted(int(x.split()[1]) for x in lines if x.startswith("prefill "))
                    self.assertEqual(seeded, [1, 1, 11, 11, 64], lines)
                    steps = [x for x in lines if x.startswith("step ")]
                    expected_steps = sum(
                        maximum - 1 if length <= 64 else length + maximum - 1
                        for length, maximum in requests
                    )
                    self.assertEqual(len(steps), expected_steps, lines)

    def test_context_is_validated_before_allocating_state(self):
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                with running_server(
                    ids=[self.output_id], flags=("--batched-prefill",),
                    continuous=continuous,
                ) as (url, _, trace):
                    status, raw = post(url, self.message(self.prompt_for_length(64),
                                                         194, stream=True))
                    self.assertEqual(status, 400, raw)  # 64 + 194 - 1 > 256
                    self.assertEqual(trace_lines(trace), [])

    def test_forbidden_bf16_env_rejected_without_startup(self):
        env = os.environ.copy()
        env["HERMES_PP2_PREFILL_BF16"] = "1"
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                command = [ARGS.server, "--batched-prefill"]
                if continuous:
                    command.append("--continuous-scheduler")
                result = subprocess.run(command, env=env, text=True, capture_output=True,
                                        timeout=8)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("fail-closed", result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[__file__, *OTHER], verbosity=2)
