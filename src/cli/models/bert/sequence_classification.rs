use std::collections::HashMap;

use lazy_static::lazy_static;

use crate::cli::{models::bert::Model, pipelines::Pipeline};

/// Text Classification
/// -------------------

/// Available models to use with Bert for Text Classification
pub static TEXT_CLASSIFICATION_MODELS: &[Model; 2] =
    &[Model::new_base_uncased(), Model::new_base_cased()];

/// The default model to use
pub static DEFAULT_TEXT_CLASSIFICATION_MODEL: Model = Model::new_base_uncased();

lazy_static! {
    /// Available models for each pipeline
    pub static ref MODELS_BY_PIPELINE: HashMap<Pipeline, &'static [Model; 2]> =
        [(Pipeline::TextClassification, TEXT_CLASSIFICATION_MODELS)]
            .iter()
            .copied()
            .collect();

    /// Default model for each pipeline
    pub static ref DEFAULT_MODEL_BY_PIPELINE: HashMap<Pipeline, Model> =
        [(Pipeline::TextClassification, DEFAULT_TEXT_CLASSIFICATION_MODEL)]
            .iter()
            .copied()
            .collect();
}
