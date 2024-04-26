use std::collections::HashMap;

use lazy_static::lazy_static;

use crate::pipelines::Pipeline;

use self::sequence_classification::text_classification;

/// BERT for Sequence Classification
pub mod sequence_classification;

/// The base model name
pub static BASE_MODEL: &str = "bert";

/// bert-base-uncased
pub static BERT_BASE_UNCASED: &str = "bert-base-uncased";

/// bert-base-cased
pub static BERT_BASE_CASED: &str = "bert-base-cased";

/// Available models to use with Bert for Sequence Classification
pub static MODELS: [&str; 2] = [BERT_BASE_UNCASED, BERT_BASE_CASED];

lazy_static! {
    /// Available models for each pipeline
    pub static ref MODELS_BY_PIPELINE: HashMap<Pipeline, &'static [&'static str]> =
        [(Pipeline::TextClassification, text_classification::MODELS)]
            .iter()
            .copied()
            .collect();

    /// Default model for each pipeline
    pub static ref DEFAULT_MODEL_BY_PIPELINE: HashMap<Pipeline, &'static str> =
        [(Pipeline::TextClassification, text_classification::DEFAULT_MODEL)]
            .iter()
            .copied()
            .collect();
}
