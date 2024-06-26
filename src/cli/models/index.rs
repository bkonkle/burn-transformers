use std::fmt::Display;

use crate::cli::pipelines::Pipeline;

use super::bert;

/// Available Models
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Model {
    /// The BERT family of models, with the specific model name contained within
    Bert(String),
}

impl Model {
    /// Get the model type
    pub fn model_type(&self) -> &str {
        match self {
            Model::Bert(_) => "bert",
        }
    }

    /// Check if the model is valid for the given pipeline
    pub fn is_supported(&self, pipeline: &Pipeline) -> bool {
        match self {
            Model::Bert(model_name) => match pipeline {
                Pipeline::TextClassification => {
                    bert::TEXT_CLASSIFICATION_MODELS.contains(&model_name.as_str())
                }
                Pipeline::TokenClassification => {
                    bert::TOKEN_CLASSIFICATION_MODELS.contains(&model_name.as_str())
                }
            },
        }
    }
}

impl TryFrom<&str> for Model {
    type Error = ModelError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if bert::ALL_MODELS.contains(&value) {
            Ok(Model::Bert(value.to_string()))
        } else {
            Err(ModelError::Unknown(value.to_string()))
        }
    }
}

impl Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Model::Bert(name) = self;

        write!(f, "{}", name)
    }
}

/// Pipeline Error
#[derive(thiserror::Error, Debug)]
pub enum ModelError {
    /// No model found for the given string
    #[error("no model found for {0}")]
    Unknown(String),
}
