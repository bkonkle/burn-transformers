# 🔥 Burn Transformers

`burn-transformers`

[![Crates.io](https://img.shields.io/crates/v/burn-transformers.svg)](https://crates.io/crates/burn-transformers)
[![Docs.rs](https://docs.rs/burn-transformers/badge.svg)](https://docs.rs/burn-transformers)
[![CI](https://github.com/bkonkle/burn-transformers/workflows/CI/badge.svg)](https://github.com/bkonkle/burn-transformers/actions)
[![Coverage Status](https://coveralls.io/repos/github/bkonkle/burn-transformers/badge.svg?branch=main)](https://coveralls.io/github/bkonkle/burn-transformers?branch=main)

## State-of-the-art Machine Learning for the Burn deep learning library in Rust

Burn Transformers is an early-stage work-in-progress port of the [Huggingface Transformers](https://huggingface.co/docs/transformers/index) library from Python to Rust. It endeavors to provide many pretrained models to perform tasks on different modalities such as text, vision, and audio.

🔥 Transformers provides APIs to quickly download and use those pretrained models on a given input, taking advantage of the great tools Huggingface has provided as part of [Candle](https://github.com/huggingface/candle) - their minimalist ML framework for Rust - and fine-tune them on your own datasets using the excellent [Burn](https://burn.dev/) deep learning library in Rust.

⚠️ Alpha Disclaimer
NOTE: This library is in early development, and the API may shift rapidly as it evolves. Be advised that this is not yet recommended for Production use.

## Features

| Pipeline | Model | Status |
| --- | --- | --- |
| Text Classification | BERT | ✅ |
| | RoBERTa | ❌ |
| | ALBERT | ❌ |
| | DistilBERT | ❌ |
| | Others... | ❌ |
| Token Classification | ... | ❌ |
| Zero Shot Classification | ... | ❌ |
| Text Generation | ... | ❌ |
| Summarization | ... | ❌ |
| Image / Video / Audio Classification | ... | ❌ |
| Question Answering | ... | ❌ |
| Others... | ... | ❌ |

## Installation

### Cargo

Install Rust and Cargo by following [this guide](https://www.rust-lang.org/tools/install).

Add this to your Cargo.toml:

```toml
    [dependencies]
    bert-transformers = { git = "https://github.com/bkonkle/burn-transformers" }
```

Once `bert-burn` is published to crates.io, this project will be able to be published there as well.

## Development

To set up a development environment to build this project, first install some helpful tools.

### Clippy

For helpful linting rools, install [Clippy](https://github.com/rust-lang/rust-clippy)

Run it with `cargo`:

```sh
cargo clippy --fix
```

If you're using VS Code, configure the `rust-analyzer` plugin to use it (in _settings.json_):

```json
{
    "rust-analyzer.checkOnSave.command": "clippy"
}
```

### pre-commit

Install pre-commit to automatically set up Git hook scripts.

In Ubuntu, the package to install is `pre-commit`:

```sh
sudo apt install pre-commit
```

On Mac with Homebrew, the package is also `pre-commit`:

```sh
brew install pre-commit
```

### libclang

The `cargo-spellcheck` utility depends on [`libclang`](https://clang.llvm.org/doxygen/group__CINDEX.html).

In Ubuntu, the package to install is `libclang-dev`:

```sh
sudo apt install libclang-dev
```

### Cargo Make

To use build scripts from the _Makefile.toml_, install Cargo Make:

```sh
cargo install cargo-make
```

Run "setup" to install some tooling dependencies:

```sh
cargo make setup
```

### Update Dependencies

First, install the `outdated` command for `cargo`:

```sh
cargo install cargo-outdated
```

Then, update and check for any major dependency changes:

```sh
cargo update
cargo outdated
```

## License

Licensed under the MIT license ([LICENSE](LICENSE) or <http://opensource.org/licenses/MIT>).
