use crate::models::Model;

/// Text Classification
pub mod text_classification;

/// Available Pipelines
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Pipeline {
    /// Text Classification
    TextClassification,
}

impl Pipeline {
    /// Get the unique string token that identifies this pipeline
    pub fn as_str(&self) -> &str {
        match self {
            Pipeline::TextClassification => text_classification::PIPELINE,
        }
    }
}

impl TryFrom<String> for Pipeline {
    type Error = PipelineError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value == text_classification::PIPELINE {
            Ok(Pipeline::TextClassification)
        } else {
            Err(PipelineError::Unknown(value))
        }
    }
}

impl Pipeline {
    /// Get the available models for this pipeline
    pub fn default_model(&self) -> Model {
        match self {
            Pipeline::TextClassification => text_classification::DEFAULT_MODEL,
        }
    }
}

/// Pipeline Error
#[derive(thiserror::Error, Debug)]
pub enum PipelineError {
    /// No pipeline found for the given string
    #[error("no pipeline found for {0}")]
    Unknown(String),
}
