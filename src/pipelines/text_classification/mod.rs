/// The Text Classification Pipeline
pub mod pipeline;

/// Batcher
pub mod batcher;

/// Training
pub mod training;

/// Inference
pub mod inference;

pub use batcher::Batcher;
pub use inference::infer;
pub use training::train;
