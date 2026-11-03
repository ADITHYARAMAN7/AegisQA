"""
Unit & Integration Tests for Hybrid Okapi BM25 + TF-IDF Retrieval and Ragas/DeepEval Mathematical Metrics.
"""
import json
import os
import unittest
from aegisqa.retrieval.hybrid_retriever import HybridBM25Retriever
from aegisqa.agent.omnihome_agent import OmniHomeSmartAgent
from aegisqa.evaluators.metrics_engine import GroundednessEvaluator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestHybridBM25AndRAGEvaluators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT_DIR, "datasets", "golden_dataset.json"), "r", encoding="utf-8") as f:
            cls.golden_data = json.load(f)
        cls.retriever = HybridBM25Retriever()
        cls.evaluator = GroundednessEvaluator(groundedness_threshold=0.85, max_hallucination_rate=0.05)

    def test_01_okapi_bm25_and_tfidf_cosine_ranking(self):
        chunks = self.retriever.retrieve("My AeroSense Pro fan won't connect to 5GHz Wi-Fi. What is its wattage?", top_k=2)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].doc_id, "DOC-FAN-001")
        self.assertGreater(chunks[0].bm25_score, 3.0)
        self.assertGreater(chunks[0].tfidf_cosine, 0.15)

    def test_02_average_precision_at_k_math(self):
        self.assertEqual(self.evaluator.compute_average_precision_at_k("DOC-FAN-001", ["DOC-FAN-001", "DOC-RO-003"]), 1.0)
        self.assertEqual(self.evaluator.compute_average_precision_at_k("DOC-FAN-001", ["DOC-RO-003", "DOC-FAN-001"]), 0.5)
        self.assertEqual(self.evaluator.compute_average_precision_at_k(None, []), 1.0)
        self.assertEqual(self.evaluator.compute_average_precision_at_k(None, ["DOC-FAN-001"]), 0.0)

    def test_03_golden_rag_groundedness_and_hallucination_suite(self):
        agent = OmniHomeSmartAgent(inject_regression=False)
        rag_cases = [c for c in self.golden_data if c["category"] in ("RAG_GROUNDEDNESS", "HALLUCINATION_TRAP")]
        for case in rag_cases:
            with self.subTest(case_id=case["id"]):
                resp = agent.respond(case["query"])
                scores = self.evaluator.evaluate(
                    answer=resp.answer,
                    retrieved_contexts=resp.retrieved_contexts,
                    expected_facts=case["expected_facts"],
                    forbidden_claims=case["forbidden_claims"],
                    expected_doc_id=case["expected_doc_id"],
                    retrieved_doc_ids=resp.retrieved_doc_ids,
                    query=case["query"],
                )
                self.assertTrue(scores["groundedness"].passed, scores["groundedness"].reason)
                self.assertTrue(scores["hallucination_rate"].passed, scores["hallucination_rate"].reason)
                self.assertTrue(scores["contextual_precision"].passed, scores["contextual_precision"].reason)
                self.assertTrue(scores["answer_relevancy"].passed, scores["answer_relevancy"].reason)

    def test_04_injected_regression_caught_by_claim_verifier(self):
        buggy_agent = OmniHomeSmartAgent(inject_regression=True)
        fan_case = next(c for c in self.golden_data if c["id"] == "GOLD-RAG-01")
        resp = buggy_agent.respond(fan_case["query"])
        scores = self.evaluator.evaluate(
            answer=resp.answer,
            retrieved_contexts=resp.retrieved_contexts,
            expected_facts=fan_case["expected_facts"],
            forbidden_claims=fan_case["forbidden_claims"],
            expected_doc_id=fan_case["expected_doc_id"],
            retrieved_doc_ids=resp.retrieved_doc_ids,
            query=fan_case["query"],
        )
        self.assertFalse(scores["groundedness"].passed)
        self.assertFalse(scores["hallucination_rate"].passed)

    def test_05_hallucination_trap_abstention_on_unknown_product(self):
        agent = OmniHomeSmartAgent(inject_regression=False)
        trap_case = next(c for c in self.golden_data if c["id"] == "GOLD-RAG-05")
        resp = agent.respond(trap_case["query"])
        self.assertEqual(len(resp.retrieved_doc_ids), 0)
        self.assertIn("not in our catalog", resp.answer)
