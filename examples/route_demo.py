"""Synthetic CPU illustration only: no model, benchmark, or new experiment.

Run from the repository root: ``python examples/route_demo.py``.
Scores and thresholds below are illustrative fixtures, not trained predictions
or frozen calibration assets from a reported experiment.
"""

import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routing.schedule import ANCHORS, MODES, quantize_score, route_step, uniform


def simulate(mode, scores, probabilities, threshold=-3.0, tie_probability=0.5):
    """Illustrate ten decisions; neither cost matching nor GPU work is simulated."""
    decisions = []
    streak = 0
    for step in range(10):
        raw_score = None if step in ANCHORS else scores[step]
        u = uniform("eval", 0, 0, 0, 0, 0, step,
                    salt=0 if mode == "content60" else 1)
        internal, reason = route_step(
            mode, step, raw_score, u,
            threshold_key=quantize_score(threshold),
            tie_probability=tie_probability,
            probability=probabilities[step], streak=streak,
        )
        streak = streak + 1 if internal else 0
        decisions.append({
            "step": step, "path": "shallow" if internal else "Full",
            "supplied_score": raw_score, "reason": reason,
        })
    return {
        "mode": mode,
        "shallow_steps": sum(row["path"] == "shallow" for row in decisions),
        "total_steps": 10,
        "steps": decisions,
    }


def main():
    scores = [None, -4.2, -3.4, -3.0, None, -3.8, -1.4, -4.6, -2.7, None]
    probabilities = [0.0, 0.9, 0.9, 0.8, 0.0, 0.7, 0.4, 0.2, 0.3, 0.0]
    report = {
        "kind": "synthetic_protocol_demo",
        "is_experimental_result": False,
        "notice": (
            "Illustrative scores/probabilities only. No FastWAM, real hidden MLP, "
            "environment, training, timing, or equal-compute claim. content60's "
            "name does not enforce six shallow decisions per inference."
        ),
        "examples": [simulate(mode, scores, probabilities) for mode in MODES],
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
