"""Validate and summarize archived aggregate results without ML dependencies.

This script regenerates tables from included aggregates. It does not execute
FastWAM, simulate LIBERO, or independently audit absent raw evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "summary.json"
DEFAULT_PROVENANCE = ROOT / "results" / "provenance.json"


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _number(value, name, lower=0.0, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < lower:
        raise ValueError(f"{name} must be finite and >= {lower}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be <= {upper}")


def _success_row(row, name):
    _count(row["episodes"], f"{name}.episodes")
    _count(row["successes"], f"{name}.successes")
    if row["episodes"] == 0 or row["successes"] > row["episodes"]:
        raise ValueError(f"{name} has invalid success counts")
    _number(row["success_rate"], f"{name}.success_rate", upper=1.0)
    if not math.isclose(row["success_rate"], row["successes"] / row["episodes"], abs_tol=1e-12):
        raise ValueError(f"{name} success rate disagrees with counts")
    if "closed_loop_shallow_ratio" in row:
        _number(row["closed_loop_shallow_ratio"], f"{name}.closed_loop_shallow_ratio", upper=1.0)


def validate_summary(data):
    """Check count arithmetic, metric meaning, and within-run baseline identity."""
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported result schema")
    if not isinstance(data.get("runs"), list) or not data["runs"]:
        raise ValueError("At least one experiment run is required")
    ids = [run["id"] for run in data["runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Run IDs must be unique")
    for run in data["runs"]:
        name = run["id"]
        counts = run["counts"]
        for field in ("episodes", "new_episodes", "reused_episodes", "timing_calls"):
            _count(counts[field], f"{name}.{field}")
        if counts["episodes"] != counts["new_episodes"] + counts["reused_episodes"]:
            raise ValueError(f"{name}: new and reused episodes do not sum to scored episodes")
        if not run.get("limitations") or not run.get("source_ids"):
            raise ValueError(f"{name}: limitations and source IDs are required")
        if run["kind"] == "closed_loop_and_timing":
            policies = run["policies"]
            modes = [p["mode"] for p in policies]
            if len(modes) != len(set(modes)) or modes.count("full") != 1:
                raise ValueError(f"{name}: exactly one Full baseline and unique policies are required")
            if sum(p["episodes"] for p in policies) != counts["episodes"]:
                raise ValueError(f"{name}: policy episode counts disagree with run count")
            for row in policies:
                label = f"{name}.{row['mode']}"
                _success_row(row, label)
                for key in ("infer_ms", "denoise_ms"):
                    _number(row[key], f"{label}.{key}", lower=1e-12)
                suites = row["per_suite"].values()
                if sum(s["episodes"] for s in suites) != row["episodes"]:
                    raise ValueError(f"{label}: suite episode counts disagree")
                if sum(s["successes"] for s in row["per_suite"].values()) != row["successes"]:
                    raise ValueError(f"{label}: suite success counts disagree")
                for suite, vals in row["per_suite"].items():
                    _count(vals["successes"], f"{label}.{suite}.successes")
                    _count(vals["episodes"], f"{label}.{suite}.episodes")
                    if vals["successes"] > vals["episodes"]:
                        raise ValueError(f"{label}.{suite}: successes exceed episodes")
        elif run["kind"] == "timing_only":
            if counts["episodes"] != 0 or "policies" in run:
                raise ValueError(f"{name}: timing-only runs cannot count parent episodes as new evaluations")
            rows = run["timings"]
            if sum(row["calls"] for row in rows) != counts["timing_calls"]:
                raise ValueError(f"{name}: timing-call counts disagree")
            keys = [(row["suite"], row["mode"]) for row in rows]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{name}: duplicate timing group")
            for suite in {row["suite"] for row in rows}:
                if sum(row["suite"] == suite and row["mode"] == "full" for row in rows) != 1:
                    raise ValueError(f"{name}.{suite}: exactly one same-suite Full baseline is required")
            for row in rows:
                label = f"{name}.{row['suite']}.{row['mode']}"
                _count(row["calls"], f"{label}.calls")
                for key in ("infer_ms", "denoise_ms"):
                    _number(row[key], f"{label}.{key}", lower=1e-12)
                _number(row["timing_shallow_ratio"], f"{label}.timing_shallow_ratio", upper=1.0)
            for row in run.get("parent_background", {}).get("policies", []):
                _success_row(row, f"{name}.parent.{row['mode']}")
        else:
            raise ValueError(f"{name}: unknown experiment kind {run['kind']!r}")
    return data


def load_results(path=DEFAULT_RESULTS):
    with Path(path).open(encoding="utf-8") as handle:
        return validate_summary(json.load(handle))


def verify_provenance(path=DEFAULT_RESULTS, provenance_path=DEFAULT_PROVENANCE):
    """Verify the included aggregate file, not the absent archived originals."""
    path = Path(path)
    with Path(provenance_path).open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    if provenance.get("schema_version") != 1:
        raise ValueError("Unsupported provenance schema")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != provenance["public_dataset_sha256"]:
        raise ValueError("Aggregate dataset SHA256 disagrees with provenance.json")
    data = load_results(path)
    sources = provenance["sources"]
    source_ids = [item["id"] for item in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Provenance source IDs must be unique")
    for item in sources:
        digest = item["sha256"]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("Invalid archived source SHA256")
    if not all(set(run["source_ids"]) <= set(source_ids) for run in data["runs"]):
        raise ValueError("A run references an unknown provenance source")
    return provenance


def markdown_summary(data):
    """Render each evaluation separately; never pool independent baselines."""
    lines = ["# Archived experiment aggregates", "", data["timing_definition"], ""]
    for run in data["runs"]:
        lines.extend([f"## {run['title']}", ""])
        counts = run["counts"]
        lines.extend([
            f"Scored episodes: {counts['episodes']} ({counts['new_episodes']} new, "
            f"{counts['reused_episodes']} reused). Formal timing calls: {counts['timing_calls']}.", ""
        ])
        if run["kind"] == "closed_loop_and_timing":
            baseline = next(p for p in run["policies"] if p["mode"] == "full")
            lines.extend([
                "| Policy | Successes | Success rate | Full inference (ms) | Denoising (ms) | Full inference speedup | Closed-loop shallow rate |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for row in run["policies"]:
                ratio = row.get("closed_loop_shallow_ratio")
                ratio_text = "not exported" if ratio is None else f"{100 * ratio:.2f}%"
                lines.append(
                    f"| {row['mode']} | {row['successes']}/{row['episodes']} | "
                    f"{100 * row['success_rate']:.2f}% | {row['infer_ms']:.2f} | "
                    f"{row['denoise_ms']:.2f} | {baseline['infer_ms'] / row['infer_ms']:.3f}x | {ratio_text} |"
                )
        else:
            lines.extend([
                "| Suite | Policy | Calls | Full inference (ms) | Denoising (ms) | Same-suite Full speedup | Timing shallow rate |",
                "|---|---|---:|---:|---:|---:|---:|",
            ])
            baselines = {row["suite"]: row for row in run["timings"] if row["mode"] == "full"}
            for row in run["timings"]:
                lines.append(
                    f"| {row['suite']} | {row['mode']} | {row['calls']} | "
                    f"{row['infer_ms']:.2f} | {row['denoise_ms']:.2f} | "
                    f"{baselines[row['suite']]['infer_ms'] / row['infer_ms']:.3f}x | "
                    f"{100 * row['timing_shallow_ratio']:.2f}% |"
                )
            flags = run["reported_audits"]
            lines.extend(["", f"Parent pairing audit passed: **{str(flags['parent_pairing_passed']).lower()}**."])
        lines.extend(["", "Limitations:", ""])
        lines.extend(f"- {item}" for item in run["limitations"])
        lines.append("")
    lines.append("These tables regenerate archived aggregates; they do not execute or reproduce policy evaluation.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--json", action="store_true", help="Print validated JSON instead of Markdown")
    parser.add_argument("--verify-only", action="store_true", help="Validate the aggregates and their included SHA256")
    args = parser.parse_args(argv)
    data = load_results(args.input)
    verify_provenance(args.input, args.provenance)
    if args.verify_only:
        print(f"Verified {len(data['runs'])} aggregate runs and the included dataset SHA256; raw originals are not included.")
    elif args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(markdown_summary(data), end="")


if __name__ == "__main__":
    main()
