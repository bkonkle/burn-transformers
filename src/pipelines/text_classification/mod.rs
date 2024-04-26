/// Common model config and traits for text classification
pub mod model;

/// Batcher
pub mod batcher;

/// Training
pub mod training;

/// Inference
pub mod inference;

pub use batcher::Batcher;
pub use inference::infer;
pub use model::{Config, Model, ModelConfig};
pub use training::train;

use crate::models::{self, bert};

/// The unique string token that identifies this pipeline
pub static PIPELINE: &str = "text-classification";

/// The default model to use for text classification
pub static DEFAULT_MODEL: models::Model = models::Model::Bert(bert::BERT_BASE_UNCASED);
