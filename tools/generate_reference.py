"""Regenerate committed Python fixtures; not needed to run the Elixir tests.

Run: python tools/generate_reference.py
Optional development dependencies: numpy==2.3.5, einops==0.8.2.
Only the common subset of the two libraries is used. Nx-specific behavior
(dtype promotion and vectorization) is covered by the native tests.
"""

from __future__ import annotations

import argparse
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
    if isinstance(value, dict):
        return "%{" + ", ".join(
            f"{elixir(key)} => {elixir(item)}" for key, item in value.items()
        ) + "}"
    if isinstance(value, (int, float)):
        text = repr(value)
        if "e" in text and "." not in text.split("e")[0]:
            text = text.replace("e", ".0e")
        return text
    raise TypeError(f"Unsupported fixture value: {type(value).__name__}")


def build_transform_cases() -> list[dict[str, Any]]:
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

    return cases


def build_boolean_cases() -> list[dict[str, Any]]:
    # Python also supports any/all. Keep these supplemental fixtures separate
    # so the original 67 numerical references remain unchanged.
    specifications = [case for case in build_transform_cases()
                      if case["operation"] == "reduce" and case["reduction"] == "sum"]
    cases = []
    for reduction in ["any", "all"]:
        for index, case in enumerate(specifications):
            shape = case["shape"]
            values = ((np.arange(int(np.prod(shape))) * 7) % 11 - 5).astype(np.float32)
            x = values.reshape(shape)
            if index in (0, 1):
                # Include all-zero, all-nonzero, and mixed groups in each
                # permutation case. Different batches have different masks.
                x[0, :, 0], x[0, :, 1] = 0, -2
                x[1, :, 0], x[1, :, 1] = 3, 0
                x[0, 0, 2], x[0, 1:, 2] = 0, 2
            elif index == 2:
                # Exercise scalar false from any and scalar true from all.
                x.fill(0 if reduction == "any" else -2)
            elif index == 3:
                x[0], x[1] = 0, -2
            else:
                x[0, :2, :3, 0] = 0
                x[0, :2, 3:, 0] = -2
                x[0, 2:, :3, 0] = [[0, 3, -1], [2, 0, 4]]
                x[0, 2:, 3:, 0] = 3
            result = np.asarray(einops.reduce(x, case["pattern"], reduction, **case["axes"]))
            assert result.dtype == np.bool_
            cases.append({
                "operation": "reduce", "shape": shape, "pattern": case["pattern"],
                "reduction": reduction, "axes": case["axes"], "input": values.tolist(),
                "expected_shape": tuple(result.shape),
                # Nx uses unsigned 0/1 tensors for logical results.
                "expected": result.astype(np.uint8).reshape(-1).tolist(),
            })
        assert {value for case in cases if case["reduction"] == reduction
                for value in case["expected"]} == {0, 1}
    return cases


def build_parse_shape_cases() -> list[dict[str, Any]]:
    # parse_shape has no tensor-value result. Keep its Python-returned maps
    # separate from the original 67 transform fixtures.
    specifications = [
        ((), ""),
        ((), "   "),
        ((), "..."),
        ((4,), "channels"),
        ((4,), "_"),
        ((4,), "... channels"),
        ((4,), "channels ..."),
        ((1, 1), "batch channels"),
        ((1, 1), "_ _"),
        ((2, 3, 4), "batch height channels"),
        ((2, 3, 4), "batch _ channels"),
        ((2, 3, 4), "_ _ channels"),
        ((2, 3, 4), "_ _ _"),
        ((2, 3, 4), "..."),
        ((2, 3, 4, 5), "batch ... channels"),
        ((2, 5), "batch ... channels"),
        ((2, 3, 4), "... channels"),
        ((2, 3, 4), "batch ..."),
        ((2, 3, 4, 5), "_ ... channels"),
        ((2, 3, 4, 5), "batch ... _"),
        ((2, 3, 4, 5), "... _ channels"),
        ((2, 3, 4, 5), "batch _ ..."),
        ((2, 3, 4), "Batch2 patch_h channels"),
        ((2, 3, 4), " batch  _  channels "),
    ]
    return [
        {
            "shape": shape,
            "pattern": pattern,
            "expected": einops.parse_shape(np.empty(shape, dtype=np.float32), pattern),
        }
        for shape, pattern in specifications
    ]


def write_transform_fixtures(cases: list[dict[str, Any]], destination: Path) -> None:
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
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {len(cases)} Python reference fixtures: {destination}")


def generate(output_dir: Path = ROOT / "test/fixtures") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_transform_fixtures(build_transform_cases(), output_dir / "reference_cases.exs")
    write_transform_fixtures(build_boolean_cases(), output_dir / "boolean_cases.exs")

    shape_cases = build_parse_shape_cases()
    lines = [
        f"# Generated with Python einops {einops.__version__} and NumPy {np.__version__}.",
        "# Rebuild with: python tools/generate_reference.py",
        "# Expected maps come from Python einops.parse_shape.",
        "[",
    ]
    for case in shape_cases:
        lines.extend([
            "  %{",
            f"    shape: {elixir(case['shape'])},",
            f"    pattern: {elixir(case['pattern'])},",
            f"    expected: {elixir(case['expected'])}",
            "  },",
        ])
    lines.append("]\n")
    destination = output_dir / "parse_shape_cases.exs"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {len(shape_cases)} Python parse_shape fixtures: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "test/fixtures",
        help="write fixtures here instead of replacing the committed fixtures",
    )
    generate(parser.parse_args().output_dir)
