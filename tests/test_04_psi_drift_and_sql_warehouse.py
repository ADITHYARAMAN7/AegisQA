"""
Tests for Statistical Drift Detection (Population Stability Index PSI, KL Divergence, Welch's t-test),
SQL/Athena Warehouse Compliance, and Automated Regression Test Synthesis.
"""
import os
import unittest
from aegisqa.pipeline.statistical_drift import StatisticalDriftMetrics, TelemetryDataPipelineValidator

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestPSIDriftAndSQLWarehouse(unittest.TestCase):
    def setUp(self):
        self.validator = TelemetryDataPipelineValidator(db_path=":memory:")

    def test_01_psi_and_kl_divergence_mathematical_properties(self):
        # Identical distributions must yield PSI = 0.0 and KL = 0.0
        base = [100.0 + i * 2.0 for i in range(100)]
        psi_same, kl_same = StatisticalDriftMetrics.compute_psi_and_kl(base, base)
        self.assertEqual(psi_same, 0.0)
        self.assertEqual(kl_same, 0.0)

        # Shifted distribution must yield PSI > 0.25 (critical drift)
        shifted = [180.0 + i * 2.0 for i in range(100)]
        psi_shift, _ = StatisticalDriftMetrics.compute_psi_and_kl(base, shifted)
        self.assertGreater(psi_shift, 0.25)

    def test_02_sql_warehouse_integrity_pii_and_psi_drift_gate(self):
        results = self.validator.run_all_sql_validations()
        self.assertEqual(len(results), 3)
        for r in results:
            with self.subTest(check_name=r.check_name):
                self.assertTrue(r.passed, r.summary)

    def test_03_auto_convert_production_failures_to_regression_dataset(self):
        out_path = os.path.join(ROOT_DIR, "datasets", "auto_generated_regressions.json")
        regressions = self.validator.convert_production_failures_to_regression_tests(out_path)
        self.assertGreaterEqual(len(regressions), 4)
        self.assertTrue(os.path.exists(out_path))
