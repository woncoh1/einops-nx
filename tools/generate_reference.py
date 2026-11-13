"""Regenerate committed value fixtures; not needed to run the Elixir tests.

Run: python tools/generate_reference.py
Optional development dependencies: numpy, einops.
Only the common subset of the two libraries is used. Nx-specific behavior
(dtype promotion, vectorization, any/all) is covered by the native tests.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
from typing import Any

import einops
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def elixir(value: Any) -> str:
    """Encode the limited literal types used in these trusted fixtures."""
    if isinstance(value, str):
        import json
        return json.dumps(value)
    if isinstance(value, tuple):
        return "{" + ", ".join(elixir(item) for item in value) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(elixir(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        text = repr(value)
        if "e" in text and "." not in text.split("e")[0]:
            text = text.replace("e", ".0e")
        return text
    raise TypeError(f"Unsupported fixture value: {type(value).__name__}")


def generate() -> None:
    cases: list[dict[str, Any]] = []

    def add(operation: str, shape: tuple[int, ...], pattern: str,
            reduction: str | None = None, **axes: int) -> None:
        # Signed, non-monotone values detect ordering errors and exercise
        # reductions without creating excessively large product results.
        size = int(np.prod(shape))
        values = ((np.arange(size) * 7) % 11 - 5).astype(np.float32)
        x = values.reshape(shape)
        if operation == "reduce":
            y = einops.reduce(x, pattern, reduction, **axes)
        else:
            y = getattr(einops, operation)(x, pattern, **axes)
        y = np.asarray(y)
        assert np.isfinite(y).all(), (operation, pattern)
        cases.append({"operation": operation, "shape": shape, "pattern": pattern,
                      "reduction": reduction, "axes": axes,
                      "input": values.tolist(), "expected_shape": tuple(y.shape),
                      "expected": y.reshape(-1).tolist()})

    for order in permutations(["a", "b", "c"]):
        labels = " ".join(order)
        add("rearrange", (2, 3, 4), f"a b c -> {labels}")
        add("rearrange", (2, 3, 4), f"a b c -> ({labels})")
        add("rearrange", (2, 3, 4), f"a b c -> ({order[0]} {order[1]}) {order[2]}")
        add("rearrange", (6, 4), f"(a b) c -> {labels}", a=2)

    for shape, pattern, axes in [
        ((2, 1, 3), "b 1 c -> 1 c b 1", {}),
        ((2, 3), "b (1 c) -> (b 1) c", {}),
        ((), " -> 1 1", {}),
        ((1, 1), "1 1 ->", {}),
        ((2, 3, 4), "b ... c -> c ... b", {}),
        ((2, 3, 4), "b ... -> b (...)", {}),
        ((2, 3, 4), "... c -> (c ...)", {}),
        ((2, 3), "b ... c -> c b ...", {}),
        ((2,), "b ... -> b (...)", {}),
        ((1, 4, 6, 1), "b (h ph) (w pw) c -> b (h w) (ph pw c)", {"ph": 2, "pw": 3}),
    ]:
        add("rearrange", shape, pattern, **axes)

    for shape, pattern, axes in [
        ((3,), "n -> n r", {"r": 2}),
        ((3,), "n -> (n r)", {"r": 2}),
        ((3,), "n -> (r n)", {"r": 2}),
        ((2, 3), "a b -> b r a s", {"r": 2, "s": 2}),
        ((6,), "(h w) -> (h r) w", {"h": 2, "r": 3}),
        ((2, 3), "... -> ... r 1", {"r": 2}),
        ((), " -> a b", {"a": 2, "b": 3}),
        ((2, 1), "n 1 -> r n", {"r": 2}),
    ]:
        add("repeat", shape, pattern, **axes)

    for reduction in ["sum", "mean", "min", "max", "prod"]:
        for shape, pattern, axes in [
            ((2, 3, 4), "a b c -> a c", {}),
            ((2, 3, 4), "a b c -> (c a)", {}),
            ((2, 3), "a b ->", {}),
            ((2, 3, 4), "a ... -> a 1", {}),
            ((1, 4, 6, 1), "b (h ph) (w pw) c -> b h w c", {"ph": 2, "pw": 3}),
        ]:
            add("reduce", shape, pattern, reduction, **axes)

    lines = [
        f"# Generated with Python einops {einops.__version__} and NumPy {np.__version__}.",
        "# Rebuild with: python tools/generate_reference.py",
        "# Expected values come from Python einops, not the Elixir implementation.",
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
            f"    expected_shape: {elixir(case['expected_shape'])},",
            f"    expected: {elixir(case['expected'])}",
            "  },",
        ])
    lines.append("]\n")
    destination = ROOT / "test/fixtures/reference_cases.exs"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {len(cases)} Python reference fixtures: {destination}")


if __name__ == "__main__":
    generate()
