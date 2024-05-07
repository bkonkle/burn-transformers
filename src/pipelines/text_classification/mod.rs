/// Common model config and traits for text classification
pub mod model;

/// Batcher
pub mod batcher;

/// Training
pub mod training;

/// Inference
pub mod inference;

pub use batcher::{Batcher, Item};
pub use inference::infer;
pub use model::{Config, Model, ModelConfig};
pub use training::train;
