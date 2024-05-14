/// Common model config and traits for text classification
pub mod model;

/// Text Classification Batcher
pub mod batcher;

/// Text Classification Items
pub mod item;

/// Text Classification Training
pub mod training;

/// Text Classification Inference
pub mod inference;

pub use batcher::Batcher;
pub use inference::infer;
pub use item::Item;
pub use model::{Config, Model, ModelConfig};
pub use training::train;
