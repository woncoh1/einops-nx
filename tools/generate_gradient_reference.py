"""Generate independent finite-difference gradient references for all 67 patterns.

Run: python tools/generate_gradient_reference.py
Optional development dependencies: the versions in tools/reference-requirements.txt.

The oracle differentiates a weighted sum of Python einops outputs with central
differences in float64. Inputs are distinct and positive, so min/max have unique
extrema and products stay numerically well-conditioned. Halving the step must
agree before a fixture is emitted. Nx autodiff is never used to create expected
gradients. Ties and zero-containing products have separate analytic ExUnit tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import einops
import numpy as np

from generate_reference import ROOT, build_transform_cases, elixir

STEP = 1.0e-5


def run_example(x: np.ndarray, case: dict[str, Any]) -> np.ndarray:
    if case["operation"] == "reduce":
        result = einops.reduce(x, case["pattern"], case["reduction"], **case["axes"])
    else:
        result = getattr(einops, case["operation"])(x, case["pattern"], **case["axes"])
    return np.asarray(result, dtype=np.float64)


def numerical_gradient(x: np.ndarray, weights: np.ndarray,
                       case: dict[str, Any], step: float) -> np.ndarray:
    gradient = np.empty(x.size, dtype=np.float64)
    for index in range(x.size):
        plus, minus = x.copy(), x.copy()
        plus.flat[index] += step
        minus.flat[index] -= step
        upper = np.sum(run_example(plus, case) * weights, dtype=np.float64)
        lower = np.sum(run_example(minus, case) * weights, dtype=np.float64)
        gradient[index] = (upper - lower) / (2 * step)
    return gradient


def build_gradient_cases() -> list[dict[str, Any]]:
    cases = []
    reference_cases = build_transform_cases()
    assert len(reference_cases) == 67, "Review gradient coverage when the reference corpus changes"
    for case in reference_cases:
        size = int(np.prod(case["shape"]))
        # The largest existing input has 24 elements. Multiplication by seven
        # permutes residues modulo 29, making every input value distinct.
        assert size <= 29
        values = 0.5 + ((np.arange(size) * 7) % 29) / 29.0
        x = values.reshape(case["shape"])
        output = run_example(x, case)
        weights = (1 + ((np.arange(output.size) * 5) % 19) / 19.0).reshape(output.shape)
        gradient = numerical_gradient(x, weights, case, STEP)
        refined = numerical_gradient(x, weights, case, STEP / 2)
        np.testing.assert_allclose(gradient, refined, atol=1.0e-7, rtol=1.0e-6)
        assert np.isfinite(gradient).all(), (case["operation"], case["pattern"])
        cases.append({
            "operation": case["operation"], "shape": case["shape"],
            "pattern": case["pattern"], "reduction": case["reduction"],
            "axes": case["axes"], "input": values.tolist(),
            "output_shape": tuple(output.shape), "weights": weights.reshape(-1).tolist(),
            # Keep more precision than the f32 comparison needs while avoiding
            # insignificant last-bit finite-difference noise in committed data.
            "gradient": np.round(gradient, decimals=8).tolist(),
        })
    return cases


def render(cases: list[dict[str, Any]]) -> str:
    lines = [
        f"# Generated with Python einops {einops.__version__} and NumPy {np.__version__}.",
        "# Rebuild with: python tools/generate_gradient_reference.py",
        "# Central differences of sum(output * weights), float64, step 1.0e-5.",
        "# Each case also passes a step-halving convergence check.",
        "# Expected gradients are rounded to eight decimal places.",
        "[",
    ]
    for case in cases:
        axes = "[" + ", ".join(f"{key}: {value}" for key, value in case["axes"].items()) + "]"
        reduction = ":" + case["reduction"] if case["reduction"] else "nil"
        lines.extend([
            "  %{",
            f"    operation: :{case['operation']},",
            f"    shape: {elixir(case['shape'])},",
            f"    pattern: {elixir(case['pattern'])},",
            f"    reduction: {reduction},",
            f"    axes: {axes},",
            f"    input: {elixir(case['input'])},",
            f"    output_shape: {elixir(case['output_shape'])},",
            f"    weights: {elixir(case['weights'])},",
            f"    gradient: {elixir(case['gradient'])}",
            "  },",
        ])
    lines.append("]\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "test/fixtures",
                        help="directory for generated Elixir fixtures")
    args = parser.parse_args()
    cases = build_gradient_cases()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "gradient_cases.exs"
    destination.write_text(render(cases), encoding="utf-8")
    print(f"Generated {len(cases)} Python gradient fixtures: {destination}")


if __name__ == "__main__":
    main()
