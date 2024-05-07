use crate::cli::models::bert;

/// The unique string token that identifies this pipeline
pub static PIPELINE: &str = "text-classification";

/// The default model to use for text classification
pub static DEFAULT_MODEL: bert::Model = bert::Model::new_base_uncased();
