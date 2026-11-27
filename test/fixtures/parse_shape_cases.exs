# Generated with Python einops 0.8.2 and NumPy 2.3.5.
# Rebuild with: python tools/generate_reference.py
# Expected maps come from Python einops.parse_shape.
[
  %{
    shape: {},
    pattern: "",
    expected: %{}
  },
  %{
    shape: {},
    pattern: "   ",
    expected: %{}
  },
  %{
    shape: {},
    pattern: "...",
    expected: %{}
  },
  %{
    shape: {4},
    pattern: "channels",
    expected: %{"channels" => 4}
  },
  %{
    shape: {4},
    pattern: "_",
    expected: %{}
  },
  %{
    shape: {4},
    pattern: "... channels",
    expected: %{"channels" => 4}
  },
  %{
    shape: {4},
    pattern: "channels ...",
    expected: %{"channels" => 4}
  },
  %{
    shape: {1, 1},
    pattern: "batch channels",
    expected: %{"batch" => 1, "channels" => 1}
  },
  %{
    shape: {1, 1},
    pattern: "_ _",
    expected: %{}
  },
  %{
    shape: {2, 3, 4},
    pattern: "batch height channels",
    expected: %{"batch" => 2, "height" => 3, "channels" => 4}
  },
  %{
    shape: {2, 3, 4},
    pattern: "batch _ channels",
    expected: %{"batch" => 2, "channels" => 4}
  },
  %{
    shape: {2, 3, 4},
    pattern: "_ _ channels",
    expected: %{"channels" => 4}
  },
  %{
    shape: {2, 3, 4},
    pattern: "_ _ _",
    expected: %{}
  },
  %{
    shape: {2, 3, 4},
    pattern: "...",
    expected: %{}
  },
  %{
    shape: {2, 3, 4, 5},
    pattern: "batch ... channels",
    expected: %{"batch" => 2, "channels" => 5}
  },
  %{
    shape: {2, 5},
    pattern: "batch ... channels",
    expected: %{"batch" => 2, "channels" => 5}
  },
  %{
    shape: {2, 3, 4},
    pattern: "... channels",
    expected: %{"channels" => 4}
  },
  %{
    shape: {2, 3, 4},
    pattern: "batch ...",
    expected: %{"batch" => 2}
  },
  %{
    shape: {2, 3, 4, 5},
    pattern: "_ ... channels",
    expected: %{"channels" => 5}
  },
  %{
    shape: {2, 3, 4, 5},
    pattern: "batch ... _",
    expected: %{"batch" => 2}
  },
  %{
    shape: {2, 3, 4, 5},
    pattern: "... _ channels",
    expected: %{"channels" => 5}
  },
  %{
    shape: {2, 3, 4, 5},
    pattern: "batch _ ...",
    expected: %{"batch" => 2}
  },
  %{
    shape: {2, 3, 4},
    pattern: "Batch2 patch_h channels",
    expected: %{"Batch2" => 2, "patch_h" => 3, "channels" => 4}
  },
  %{
    shape: {2, 3, 4},
    pattern: " batch  _  channels ",
    expected: %{"batch" => 2, "channels" => 4}
  }
]
