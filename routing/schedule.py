"""CPU-accessible rules from the frozen, quota-free Task 4 v2 experiment.

Provenance: ``work/ig/task4_matched_continue/schedule.py``. The RNG key,
quantization, threshold comparison, and random comparison preserve that source.
``route_step`` extracts the anchor/streak dispatch from its ``common.py``.

These functions select a path; they neither execute FastWAM nor measure latency.
The matched control in the GPU experiment evaluated the SAME real current-hidden
MLP and called ``.item()`` before discarding the score. Passing a score into this
CPU module does not reproduce or certify that GPU-overhead control.
"""

import hashlib
import math


# Intentionally retain the parent-v2 namespace used by the authorized continuation.
RUN_ID = "20261027_task4_matched_v2"
MODEL_SEED = 42
ROUTE_SEED = 62043
MODES = ("content60", "matched_step_mlp60")
SUITES = ("libero_spatial", "libero_goal")
ANCHORS = (0, 4, 9)
ELIGIBLE = (1, 2, 3, 5, 6, 7, 8)
MAX_STREAK = 4
SCORE_SCALE = 100000
PHASE_SEEDS = {
    "calibration": 43,
    "budget": 44,
    "eval": 45,
    "timing": 45,
    "preflight": 43,
}


def uniform(phase, suite_id, task, trial, state_index, chunk, step, salt=0):
    """Original keyed RNG; independent of Python/NumPy/environment RNG states."""
    key = (
        f"{RUN_ID}/{ROUTE_SEED}/{phase}/{PHASE_SEEDS[phase]}/{suite_id}/"
        f"{task}/{trial}/{state_index}/{chunk}/{step}/{salt}"
    ).encode()
    bits = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 11
    return bits / float(1 << 53)


def quantize_score(value):
    """Original Python round-to-even rule; this is a log-gap score, not a %.

    The caller of the original runtime checked that the raw score was finite.
    ``route_step`` retains that check; this small primitive is unchanged.
    """
    return int(round(float(value) * SCORE_SCALE))


def threshold_choice(score_key, threshold_key, tie_probability, random_value):
    if score_key < threshold_key:
        return True, "predicted_gap"
    if score_key > threshold_key:
        return False, "predicted_gap"
    return random_value < tie_probability, "predicted_tie"


def random_choice(probability, random_value):
    # Deliberately no score or hidden-state argument, as in the original source.
    return random_value < probability, "matched_step_random"


def route_step(mode, step, raw_score, random_value, *, threshold_key=None,
               tie_probability=0.0, probability=None, streak=0):
    """Select one step using the original anchors, comparison, and streak rule.

    ``raw_score`` must be the already evaluated MLP output for either mode at an
    eligible step. The random control ignores its numerical value for routing.
    Anchors do not require a score. The caller resets streak for each inference
    and updates it to ``streak + 1 if internal else 0`` after each decision.

    No count quota, target-ratio correction, or forced shallow decision exists.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode!r}")
    if step not in range(10):
        raise ValueError("The frozen protocol has steps 0 through 9")
    if step in ANCHORS:
        return False, "anchor"
    if raw_score is None or not math.isfinite(float(raw_score)):
        raise ValueError("Both modes require a finite, evaluated MLP score")
    score_key = quantize_score(raw_score)
    if mode == "content60":
        if threshold_key is None:
            raise ValueError("Content routing requires its frozen threshold key")
        internal, reason = threshold_choice(
            score_key, threshold_key, tie_probability, random_value
        )
    else:
        if probability is None:
            raise ValueError("Random routing requires its frozen suite-step probability")
        internal, reason = random_choice(probability, random_value)
    if streak >= MAX_STREAK:
        internal, reason = False, "max_streak"
    return internal, reason
