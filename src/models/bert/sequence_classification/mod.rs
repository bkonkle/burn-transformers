/// Bert for Sequence Classification
pub mod model;

/// The model configuration
pub mod config;

/// Use case for text classification
pub mod text_classification;

pub use config::Config;
pub use model::{Model, ModelRecord, ModelRecordItem};
