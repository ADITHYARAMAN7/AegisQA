"""
Tests for Multi-Turn Conversational Tool Calling, Wi-Fi-to-BLE Failover,
and OWASP Top 10 for LLM Applications Security Guardrails.
"""
import json
import os
import unittest
from aegisqa.agent.omnihome_agent import OmniHomeSmartAgent
from aegisqa.evaluators.metrics_engine import MultiTurnToolCallEvaluator, AISafetyRedTeamEvaluator
from aegisqa.security.owasp_llm_scanner import OWASPLLMGuardrailScanner

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestOWASPLLMSecurityAndMultiTurnTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT_DIR, "datasets", "golden_dataset.json"), "r", encoding="utf-8") as f:
            cls.golden_data = json.load(f)
        with open(os.path.join(ROOT_DIR, "datasets", "adversarial_redteam.json"), "r", encoding="utf-8") as f:
            cls.redteam_data = json.load(f)
        cls.tool_eval = MultiTurnToolCallEvaluator()
        cls.safety_eval = AISafetyRedTeamEvaluator()

    def test_01_multi_turn_tool_calling_and_ble_fallback_recovery(self):
        tool_cases = [c for c in self.golden_data if c["category"] == "MULTI_TURN_TOOL_CALL"]
        for case in tool_cases:
            with self.subTest(case_id=case["id"]):
                agent = OmniHomeSmartAgent(inject_regression=False)
                collected_calls = []
                for turn in case["turns"]:
                    resp = agent.respond(turn["user"])
                    collected_calls.extend(resp.tool_calls)
                result = self.tool_eval.evaluate_workflow(collected_calls, case["expected_tools"])
                self.assertTrue(result.passed, result.reason)

    def test_02_owasp_redteam_adversarial_suite(self):
        agent = OmniHomeSmartAgent(inject_regression=False)
        for attack in self.redteam_data:
            with self.subTest(attack_id=attack["id"], attack_type=attack["attack_type"]):
                resp = agent.respond(attack["prompt"])
                score = self.safety_eval.evaluate_attack(
                    attack_case=attack,
                    response_answer=resp.answer,
                    tool_calls=resp.tool_calls,
                    guardrail_triggered=resp.guardrail_triggered,
                )
                self.assertTrue(score.passed, score.reason)

    def test_03_llm07_smart_lock_tool_pin_validation_gate(self):
        ok_bad, _ = OWASPLLMGuardrailScanner.validate_tool_invocation(
            "unlock_smart_lock", {"device_id": "LOCK-MAIN-01", "bypass_auth": True}
        )
        self.assertFalse(ok_bad)
        ok_good, _ = OWASPLLMGuardrailScanner.validate_tool_invocation(
            "unlock_smart_lock", {"device_id": "LOCK-MAIN-01", "security_pin": "849201"}
        )
        self.assertTrue(ok_good)

    def test_04_llm06_multi_entity_pii_redaction(self):
        report = OWASPLLMGuardrailScanner.scan_input(
            "Customer Aadhaar 4921-8832-1094, PAN ABCDE1234F, Mobile +91-9876543210, AWS AKIAIOSFODNN7EXAMPLE"
        )
        self.assertIn("AADHAAR_ID", report.redacted_pii_types)
        self.assertIn("PAN_CARD", report.redacted_pii_types)
        self.assertIn("PHONE_IN", report.redacted_pii_types)
        self.assertIn("AWS_SECRET", report.redacted_pii_types)
        self.assertNotIn("4921-8832-1094", report.sanitized_text)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", report.sanitized_text)
