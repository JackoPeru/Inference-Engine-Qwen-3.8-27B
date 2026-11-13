"""Loopback transport and credential isolation tests (all HTTP mocked)."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

from ops import manager_lease_adapter as adapter


class FakeResponse:
    status = 200

    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _n):
        return self.data


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.payload = {"protocol": "hermes-manager-lease-v1", "action": "acquire",
                        "owner": "hermes-27", "lease_id": "b" * 32,
                        "manager_instance_id": "a" * 32, "fencing_token": None}
        self.reply = {"protocol": "hermes-manager-lease-v1", "action": "acquire",
                      "lease_id": "b" * 32, "ok": True}
        self.opened = []
        self.actual_secret_file = adapter.secret_file

        class FakeOpener:
            def open(opener_self, request, timeout):
                self.opened.append((request, timeout))
                return FakeResponse(json.dumps(self.reply).encode())

        self.mock_credentials = mock.patch.object(adapter, "secret_file", side_effect=["test-manager-key", "test-lease-token"])
        self.mock_credentials.start()
        self.addCleanup(self.mock_credentials.stop)
        self.mock_opener = mock.patch.object(adapter.urllib.request, "build_opener", return_value=FakeOpener())
        self.mock_opener.start()
        self.addCleanup(self.mock_opener.stop)

    def test_loopback_request_is_post_and_auth_is_not_returned(self):
        result = adapter.call_manager("acquire", self.payload,
                                      manager_url="http://127.0.0.1:8643")
        self.assertEqual(result, self.reply)
        request, timeout = self.opened[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8643/maintenance-lease/v1/acquire")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(timeout, 8)
        self.assertEqual(request.headers["Authorization"], "Bearer test-manager-key")
        self.assertNotIn("test-manager-key", json.dumps(result))

    def test_reject_external_redirect_and_url_tricks_before_opening_network(self):
        for url in ("https://127.0.0.1:8643", "http://example.com:8643",
                    "http://127.0.0.1:8643@evil.example:8643",
                    "http://127.0.0.1:8643/admin", "http://127.0.0.1:8643/?key=1",
                    "http://localhost:8643", "http://127.0.0.1:0"):
            with self.subTest(url=url), self.assertRaises(adapter.AdapterRejected):
                adapter.call_manager("acquire", self.payload, manager_url=url)
        self.assertEqual(self.opened, [])

    def test_refuse_action_mismatch(self):
        with self.assertRaisesRegex(adapter.AdapterRejected, "protocol/action"):
            adapter.call_manager("renew", self.payload, manager_url="http://127.0.0.1:8643")

    def test_refuse_unrecognized_action(self):
        with self.assertRaises(adapter.AdapterRejected):
            adapter.call_manager("delete", self.payload)

    def test_refuse_mismatched_lease_acknowledgement(self):
        self.reply["lease_id"] = "c" * 32
        with self.assertRaisesRegex(adapter.AdapterRejected, "invalid/fenced"):
            adapter.call_manager("acquire", self.payload, manager_url="http://127.0.0.1:8643")

    def test_refuse_wrong_version_acknowledgement(self):
        self.reply["protocol"] = "v0"
        with self.assertRaises(adapter.AdapterRejected):
            adapter.call_manager("acquire", self.payload, manager_url="http://127.0.0.1:8643")

    def test_refuse_explicit_denial(self):
        self.reply["ok"] = False
        with self.assertRaises(adapter.AdapterRejected):
            adapter.call_manager("acquire", self.payload, manager_url="http://127.0.0.1:8643")

    def test_refuse_http_transport_failure_without_repeating_secrets(self):
        class FailedOpener:
            def open(self, *_args, **_kwargs):
                raise urllib.error.HTTPError("http://127.0.0.1", 503, "secret-DO-NOT-PRINT", {}, None)

        with mock.patch.object(adapter.urllib.request, "build_opener", return_value=FailedOpener()):
            with self.assertRaises(adapter.AdapterRejected) as captured:
                adapter.call_manager("acquire", self.payload, manager_url="http://127.0.0.1:8643")
        self.assertNotIn("secret-DO-NOT-PRINT", str(captured.exception))

    def test_missing_credential_configuration_denied(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(adapter.AdapterRejected):
                self.actual_secret_file("HERMES_MANAGER_API_KEY_FILE")

    def test_relative_credential_path_denied(self):
        with mock.patch.dict(os.environ, {"HERMES_MANAGER_API_KEY_FILE": "relative-secret"}):
            with self.assertRaises(adapter.AdapterRejected):
                self.actual_secret_file("HERMES_MANAGER_API_KEY_FILE")


if __name__ == "__main__":
    unittest.main()
