# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Changed

- Moved the batcher up to a shared location above the sequence classification pipelines.

### Added

- Added an initial Token Classification pipeline based on Bert for NER and other slot filling tasks.


## [0.2.0]

### Changed

- Reworked the CLI tools to more easily support pipelines, models, and datasets as the library grows.
- Updated the way that "with_pooling_layer" is passed into the bert-burn model, to work with the updated branch.

## [0.1.0]

### Added

- Added an initial Text Classification pipeline implementation based on Bert for Sequence Classification.

[0.3.0]: <https://github.com/bkonkle/burn-transformers/compare/0.2.0...0.3.0>
[0.2.0]: <https://github.com/bkonkle/burn-transformers/compare/0.1.0...0.2.0>
[0.1.0]: https://github.com/bkonkle/burn-transformers/releases/tag/0.1.0
