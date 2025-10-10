# xgrammar-rs

[![CI](https://github.com/furiosa-ai/xgrammar-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/furiosa-ai/xgrammar-rs/actions/workflows/ci.yml) [![Deploy Docs](https://github.com/furiosa-ai/xgrammar-rs/actions/workflows/docs.yml/badge.svg)](https://github.com/furiosa-ai/xgrammar-rs/actions/workflows/docs.yml)

This project provides safe and idiomatic Rust bindings for the `xgrammar` C++ library.
By wrapping the C++ implementation, this crate leverages Rust's memory safety
andguarantees while providing access to `xgrammar`'s high-performance and features
for constraint decoding.

## Prerequisites

Before building the project, ensure you have the following dependencies installed:

-   **Rust toolchain**: Install via [rustup](https://rustup.rs/).
-   **CMake**: Required to build the underlying C++ `xgrammar` library.
-   **C++ compiler**: A modern C++ compiler that supports C++17 (Clang is hightly recommended.)

## Build

The C++ `xgrammar` library is included as a submodule and will be compiled automatically as part of the build process.

To build the project, run the following command:

```bash
cargo build --release
```

This will create a release build in the `target/release` directory.

## Test

This project uses `cargo-nextest` for running tests. To execute all tests, run:

```bash
make test
```

You can also pass arguments to `cargo nextest` via the `TEST_ARGS` variable. For example, to run a specific test suite:

```bash
make test TEST_ARGS="--test test_grammar"
```
