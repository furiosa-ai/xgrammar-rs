# xgrammar-rs

This project uses [xgrammar](https://github.com/mlc-ai/xgrammar) **v0.2.3** as a submodule.

[![CI](https://github.com/furiosa-ai/xgrammar-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/furiosa-ai/xgrammar-rs/actions/workflows/ci.yml) [![Deploy Docs](https://github.com/furiosa-ai/xgrammar-rs/actions/workflows/docs.yml/badge.svg)](https://github.com/furiosa-ai/xgrammar-rs/actions/workflows/docs.yml)

This project provides safe and idiomatic Rust bindings for the `xgrammar` C++ library.
By wrapping the C++ implementation, this crate leverages Rust's memory safety
andguarantees while providing access to `xgrammar`'s high-performance and features
for constraint decoding.

## Highlights

- `Grammar`, `GrammarCompiler`, `CompiledGrammar`, `TokenizerInfo` — core API
  for compiling grammars (BNF, JSON schema, regex, structural tags) against a
  tokenizer.
- `GrammarMatcher` — token-by-token constrained decoding, including
  `is_completed()` (root-rule match without stop token), `fork()` for
  speculative / branching decoding, and `traverse_draft_tree()` for filling
  per-node token bitmasks over a speculative-decoding draft tree.
- `BatchGrammarMatcher` — batched helpers operating on a slice of matchers:
  `batch_fill_next_token_bitmask` (parallel, thread-pool-backed),
  `batch_accept_token`, `batch_accept_string`, `batch_rollback`
  (sequential static helpers).
- Serialization — `serialize_json()` / `deserialize_json()` on `Grammar`,
  `CompiledGrammar`, and `TokenizerInfo` for persisting compilation results
  (version-locked to the vendored xgrammar's serialization format).
- Typed errors — `XGrammarErr` variants mirror the upstream xgrammar exception
  types via `XGrammarError::GetType()` (`InvalidJson`, `InvalidStructuralTag`,
  `DeserializeVersion`, `DeserializeFormat`, ...), with the previous
  coarse-grained variants kept as fallbacks for untyped C++ errors.
- Linux `x86_64` and `aarch64` (arm64) are both supported.

See the [rustdoc](https://docs.rs/xgrammar/latest/xgrammar/) for detailed method-level documentation, including when a
`BatchGrammarMatcher` instance is required vs when associated functions can be
called directly.

## Prerequisites

Before building the project, ensure you have the following dependencies installed:

-   **Rust toolchain**: Install via [rustup](https://rustup.rs/).
-   **CMake**: Required to build the underlying C++ `xgrammar` library.
-   **C++ compiler**: A modern C++ compiler that supports C++17 (Clang is highly recommended.)

## Build

The C++ `xgrammar` library is included as a submodule and will be compiled automatically as part of the build process.

To build the project, run the following command:

```bash
git clone https://github.com/furiosa-ai/xgrammar-rs.git --recurse-submodules
cd xgrammar-rs
cargo build --release
```

If you have already cloned the repository without `--recurse-submodules`, initialize the submodule with:

```bash
git submodule update --init --recursive
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
