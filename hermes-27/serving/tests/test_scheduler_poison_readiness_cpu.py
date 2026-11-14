#!/usr/bin/env python3
"""CPU-only fatal-ready-pump HTTP health and admission contract.

The mock injects one post-join ready-pump exception in actual production
scheduler code. No CUDA model, physical GPUs, or live HERMES endpoints.
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request

from prod_sched_state_test import (
    ARGS, assert_single_owner, payload, post, running_server, trace_lines,
)


class FatalReadyPumpHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(ARGS.tokenizer)
        cls.token = tokenizer.encode("hello", add_special_tokens=False).ids[0]

    def test_fatal_ready_pump_marks_unhealthy_and_refuses_new_work_503(self):
        with running_server(
            ids=[self.token], continuous=True,
            flags=("--parallel-steps", "--ready-completion",
                   "--ready-callback-delivery"),
            env_extra={"HERMES_PP2_ASYNC_STREAMS": "1",
                       "HERMES_TEST_READY_PUMP_POISON_ONCE": "1"},
        ) as (url, _, trace):
            status, raw = post(url, payload(3))
            self.assertEqual(status, 500, raw)
            self.assertIn("error", json.loads(raw))

            health = urllib.request.Request(url + "/health")
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(health, timeout=10)
            self.assertEqual(caught.exception.code, 503)
            self.assertEqual(
                json.loads(caught.exception.read()), {
                    "status": "unavailable", "ready": False,
                })
            with urllib.request.urlopen(url + "/metrics", timeout=10) as response:
                values = response.read().decode("utf-8").splitlines()
            self.assertIn("hermes_server_ready 0", values)

            status, raw = post(url, payload(2))
            self.assertEqual(status, 503, raw)
            err = json.loads(raw)["error"]
            self.assertEqual(err["code"], "scheduler_stopped")
            self.assertEqual(err["type"], "server_error")
            assert_single_owner(self, trace_lines(trace), requests=1)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
