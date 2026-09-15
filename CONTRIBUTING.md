# Contributing

Use a C++20 compiler and CMake 3.20 or newer. Before opening a pull request:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Keep the core dependency-free. New tensor engines, transports, and robot
middleware integrations should be optional targets or separate adapter
packages. Every backend must document its model contract and include tests for
shape validation, failures, and numerical parity with its reference runtime.

Do not commit model weights, robot credentials, recorded private sensor data,
or generated build output.
