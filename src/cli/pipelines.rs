use std::fmt::Display;

use super::models::{bert, Model};

/// The unique string token that identifies this pipeline
pub static TEXT_CLASSIFICATION: &str = "text-classification";

/// Available Pipelines
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Pipeline {
    /// Text Classification
    TextClassification,
}

impl Pipeline {
    /// Get the default model variant for the given pipeline
    pub fn default_model(&self) -> Model {
        match self {
            Pipeline::TextClassification => {
                Model::Bert(bert::DEFAULT_TEXT_CLASSIFICATION_MODEL.to_string())
            }
        }
    }
}

impl TryFrom<&str> for Pipeline {
    type Error = PipelineError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value == TEXT_CLASSIFICATION {
            Ok(Pipeline::TextClassification)
        } else {
            Err(PipelineError::Unknown(value.to_string()))
        }
    }
}

impl Display for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Pipeline::TextClassification => TEXT_CLASSIFICATION,
        };

        write!(f, "{}", name)
    }
}

/// Pipeline Error
#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    /// No pipeline found for the given string
    #[error("no pipeline found for {0}")]
    Unknown(String),
}
