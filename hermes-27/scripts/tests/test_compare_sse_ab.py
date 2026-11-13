#!/usr/bin/env python3
"""Synthetic fail-closed checks for SSE A/B comparator; no GPU/network."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "compare_sse_ab.py"
spec = importlib.util.spec_from_file_location("compare_sse_ab", SCRIPT)
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)


def row(n, r, variant):
    return {
        "schema": "h27.sse-timing.v1", "variant": variant,
        "clients": n, "round": r, "max_tokens": 16,
        "requests": n, "completion_tokens": n*16,
        "prompt_tokens": n*47,
        "aggregate_completion_tok_per_wall_sec": 2+n/2,
        "ttfc_p50_sec": 1., "ttfc_p95_sec": 1.2,
        "request_latency_p50_sec": 2.5, "request_latency_p95_sec": 3.,
        "outputs": [
            {"case": i, "completion_tokens": 16, "finish": "length",
             "sha256": f"{r+i+1:064x}"} for i in range(n)
        ],
    }


def write_log(path, variant, *, missing=None, duplicate=False):
    records = [row(n, r, variant) for n in cmp.CLIENTS for r in cmp.ROUNDS
               if (n, r) != missing]
    if duplicate:
        records.append(records[0])
    path.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n" +
        f"SCRATCH SSE TIMING: PASS variant={variant}\n" +
        "POST-RUN GPU MEMORY:\n0, 486 MiB, 0 %\n1, 1289 MiB, 0 %\n",
        encoding="utf-8")


class SSEAComparison(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name) / "parallel.log"
        self.l = Path(self.tmp.name) / "legacy.log"
        write_log(self.p, "parallel")
        write_log(self.l, "legacy")

    def test_equal_full_outputs_pass(self):
        result = cmp.compare(cmp.parse_log(self.p, "parallel"),
                             cmp.parse_log(self.l, "legacy"))
        self.assertEqual(result["request_count_per_variant"], 30)
        self.assertTrue(result["output_hashes_identical"])
        self.assertEqual([r["clients"] for r in result["data"]], [1, 2, 4, 8])

    def test_wrong_content_sha_is_rejected(self):
        parsed = cmp.parse_log(self.p, "parallel")
        parsed[(4, 0)]["outputs"][2]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "full generated text mismatch"):
            cmp.compare(parsed, cmp.parse_log(self.l, "legacy"))

    def test_missing_round_is_rejected(self):
        write_log(self.p, "parallel", missing=(8, 1))
        with self.assertRaisesRegex(ValueError, "missing workloads"):
            cmp.parse_log(self.p, "parallel")

    def test_duplicate_round_is_rejected(self):
        write_log(self.p, "parallel", duplicate=True)
        with self.assertRaisesRegex(ValueError, "duplicate round"):
            cmp.parse_log(self.p, "parallel")

    def test_failed_restoration_is_rejected(self):
        self.p.write_text(self.p.read_text().replace("0, 486 MiB", "0, 9000 MiB"))
        with self.assertRaisesRegex(ValueError, "missing exact idle"):
            cmp.parse_log(self.p, "parallel")

    def test_bad_token_usage_is_rejected(self):
        self.p.write_text(self.p.read_text().replace('"completion_tokens": 16',
                                                    '"completion_tokens": 15', 1))
        with self.assertRaisesRegex(ValueError, "incomplete/wrong workload"):
            cmp.parse_log(self.p, "parallel")


if __name__ == "__main__":
    unittest.main()
