"""Protocol invariants; synthetic fixtures, no torch or policy inference."""

import inspect
import random
import unittest

from routing.schedule import (
    ANCHORS, ELIGIBLE, MAX_STREAK, MODES, quantize_score,
    random_choice, route_step, threshold_choice, uniform,
)


class RoutingProtocolTests(unittest.TestCase):
    def test_rng_is_independent_of_global_random_state(self):
        before = random.getstate()
        first = uniform("eval", 0, 0, 7, 7, 3, 6, 1)
        self.assertEqual(before, random.getstate())
        try:
            random.seed(987654321)
            for _ in range(20):
                random.random()
            self.assertEqual(first, uniform("eval", 0, 0, 7, 7, 3, 6, 1))
        finally:
            random.setstate(before)
        self.assertGreaterEqual(first, 0.0)
        self.assertLess(first, 1.0)
        self.assertNotEqual(first, uniform("eval", 0, 0, 7, 7, 3, 6, 0))
        self.assertNotEqual(first, uniform("budget", 0, 0, 7, 7, 3, 6, 1))

    def test_anchors_are_full_without_mlp_score(self):
        self.assertEqual(set(ANCHORS) | set(ELIGIBLE), set(range(10)))
        self.assertFalse(set(ANCHORS) & set(ELIGIBLE))
        for mode in MODES:
            for step in ANCHORS:
                self.assertEqual(route_step(mode, step, None, 0.0), (False, "anchor"))

    def test_threshold_ties_use_strict_uniform_comparison(self):
        self.assertEqual(threshold_choice(9, 10, 0.0, 0.99), (True, "predicted_gap"))
        self.assertEqual(threshold_choice(11, 10, 1.0, 0.0), (False, "predicted_gap"))
        self.assertEqual(threshold_choice(10, 10, 0.25, 0.249), (True, "predicted_tie"))
        self.assertEqual(threshold_choice(10, 10, 0.25, 0.25), (False, "predicted_tie"))

    def test_quantization_matches_original_python_round(self):
        for value in (-1.000005, -1.000015, 0.000005, 0.000015, 1.23456):
            self.assertEqual(quantize_score(value), int(round(float(value) * 100000)))

    def test_random_control_ignores_score_but_requires_evaluation(self):
        self.assertEqual(list(inspect.signature(random_choice).parameters),
                         ["probability", "random_value"])
        for u, expected in ((0.499, True), (0.5, False)):
            answers = [route_step("matched_step_mlp60", 1, score, u, probability=0.5)
                       for score in (-1000.0, -3.0, 1000.0)]
            self.assertTrue(all(answer == (expected, "matched_step_random") for answer in answers))
        with self.assertRaises(ValueError):
            route_step("matched_step_mlp60", 1, None, 0.0, probability=1.0)

    def test_no_quota_correction_and_zero_or_seven_shallow_are_possible(self):
        for score, expected in ((1.0, 0), (-1.0, 7)):
            streak = used = 0
            for step in range(10):
                internal, _ = route_step("content60", step, score, 0.5,
                                         threshold_key=0, streak=streak)
                used += int(internal)
                streak = streak + 1 if internal else 0
            self.assertEqual(used, expected)

    def test_streak_guard_is_preserved(self):
        for mode in MODES:
            result = route_step(mode, 8, -10.0, 0.0, threshold_key=0,
                                probability=1.0, streak=MAX_STREAK)
            self.assertEqual(result, (False, "max_streak"))

    def test_nonfinite_scores_are_not_valid_risk_predictions(self):
        for mode in MODES:
            for bad in (float("nan"), float("inf"), -float("inf")):
                with self.assertRaises(ValueError):
                    route_step(mode, 1, bad, 0.5, threshold_key=0, probability=0.5)


if __name__ == "__main__":
    unittest.main()
