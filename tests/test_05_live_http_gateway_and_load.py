"""
Tests for Live Multi-Threaded TCP HTTP Gateway (`ThreadingHTTPServer`),
HMAC-SHA256 JWT Authentication, OpenAPI 3.1 Contract Validation, and Real-Socket Concurrent Load Testing.
"""
import json
import unittest
import urllib.error
import urllib.request
from aegisqa.server.http_gateway import JWTAuthManager, OmniHomeAPIGatewaySimulator


class TestLiveHTTPGatewayAndSocketLoad(unittest.TestCase):
    def setUp(self):
        self.gateway = OmniHomeAPIGatewaySimulator()

    def test_01_hmac_sha256_jwt_signing_and_tamper_rejection(self):
        token = JWTAuthManager.create_token("sdet-runner", exp_offset_sec=600)
        ok, _ = JWTAuthManager.verify_header(f"Bearer {token}")
        self.assertTrue(ok)

        # Tamper with signature byte
        tampered = token[:-2] + ("A" if token[-1] != "A" else "B")
        ok_bad, msg = JWTAuthManager.verify_header(f"Bearer {tampered}")
        self.assertFalse(ok_bad)
        self.assertIn("signature", msg)

    def test_02_live_tcp_http_server_healthz_openapi_and_chat(self):
        server, base_url = self.gateway.start_live_http_server()
        token = JWTAuthManager.create_token("integration-test")
        try:
            # 1. GET /healthz
            with urllib.request.urlopen(f"{base_url}/healthz", timeout=5) as r1:
                self.assertEqual(r1.status, 200)
                h_body = json.loads(r1.read().decode("utf-8"))
                self.assertEqual(h_body["status"], "healthy")

            # 2. GET /openapi.json
            with urllib.request.urlopen(f"{base_url}/openapi.json", timeout=5) as r2:
                self.assertEqual(r2.status, 200)
                spec = json.loads(r2.read().decode("utf-8"))
                self.assertEqual(spec["openapi"], "3.1.0")

            # 3. POST /v1/ai/chat over real TCP socket
            req = urllib.request.Request(
                f"{base_url}/v1/ai/chat",
                data=json.dumps({"query": "How do I pair my AeroSense Pro fan?"}).encode("utf-8"),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r3:
                self.assertEqual(r3.status, 200)
                chat_body = json.loads(r3.read().decode("utf-8"))
                self.assertIn("DOC-FAN-001", chat_body["retrieved_doc_ids"])
        finally:
            server.shutdown()
            server.server_close()

    def test_03_unauthorized_and_malformed_requests_rejected(self):
        r401 = self.gateway.post_ai_chat({"query": "Unlock door"}, auth_header="Bearer expired-token")
        self.assertEqual(r401.status_code, 401)
        r422 = self.gateway.post_ai_chat({"query": ""})
        self.assertEqual(r422.status_code, 422)

    def test_04_real_tcp_socket_concurrent_load_benchmark(self):
        bench = self.gateway.run_load_test(total_requests=60, concurrency=10)
        self.assertEqual(bench["transport"], "REAL_TCP_HTTP_SOCKET")
        self.assertEqual(bench["success_rate_pct"], 100.0)
        self.assertTrue(bench["sla_p95_under_250ms_passed"])
