"""Protect the scientific boundaries of the public aggregate dataset."""

import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("summarize_results", ROOT / "scripts" / "summarize_results.py")
results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(results)


class ResultIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.data = results.load_results()
        self.by_id = {run["id"]: run for run in self.data["runs"]}

    def test_included_dataset_matches_provenance(self):
        provenance = results.verify_provenance()
        self.assertTrue(all(not s["included_in_public_repository"] for s in provenance["sources"]))

    def test_different_full_baselines_are_kept_separate(self):
        fixed = self.by_id["fixed_structure_40"]["policies"][0]
        dynamic = self.by_id["dynamic_routing_40"]["policies"][0]
        self.assertEqual((fixed["successes"], dynamic["successes"]), (386, 391))
        self.assertNotEqual(fixed["infer_ms"], dynamic["infer_ms"])
        report = results.markdown_summary(self.data)
        self.assertIn("388/400", report)
        self.assertIn("387/400", report)

    def test_reused_results_are_not_counted_as_fresh(self):
        counts = self.by_id["dynamic_routing_40"]["counts"]
        self.assertEqual((counts["episodes"], counts["new_episodes"], counts["reused_episodes"]), (1200, 900, 300))
        bad = copy.deepcopy(self.data)
        bad["runs"][1]["counts"]["new_episodes"] = 1200
        with self.assertRaisesRegex(ValueError, "do not sum"):
            results.validate_summary(bad)

    def test_timing_continuation_preserves_parent_failure(self):
        plus = self.by_id["plus_timing_continuation"]
        self.assertEqual(plus["counts"]["episodes"], 0)
        self.assertEqual(plus["counts"]["parent_episodes"], 1497)
        self.assertEqual(plus["counts"]["timing_calls"], 600)
        self.assertFalse(plus["reported_audits"]["parent_pairing_passed"])
        self.assertFalse(plus["reported_audits"]["parent_original_protocol_complete"])
        self.assertFalse(plus["reported_audits"]["actual_compute_matched"])
        self.assertTrue(plus["reported_audits"]["timing_audit_passed"])

    def test_report_separates_timing_and_rollout_shallow_rates(self):
        report = results.markdown_summary(self.data)
        self.assertIn("Closed-loop shallow rate", report)
        self.assertIn("Timing shallow rate", report)
        self.assertIn("Parent pairing audit passed: **false**", report)

    def test_rejects_inconsistent_success_counts(self):
        bad = copy.deepcopy(self.data)
        bad["runs"][0]["policies"][0]["successes"] += 1
        with self.assertRaisesRegex(ValueError, "success rate disagrees"):
            results.validate_summary(bad)

    def test_rejects_nonfinite_latency(self):
        bad = copy.deepcopy(self.data)
        bad["runs"][0]["policies"][0]["infer_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            results.validate_summary(bad)

    def test_detects_changed_aggregate_bytes(self):
        changed = results.DEFAULT_RESULTS.read_bytes() + b"\n"
        with patch.object(Path, "read_bytes", return_value=changed):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                results.verify_provenance()


if __name__ == "__main__":
    unittest.main()
