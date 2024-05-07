use std::fmt::Display;

use crate::datasets::snips;

/// The Dataset enum
pub enum Dataset {
    /// Snips dataset
    Snips,
}

impl TryFrom<&str> for Dataset {
    type Error = DatasetError;

    /// Try to convert a string to a Dataset
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        if value.to_lowercase() == snips::DATASET {
            Ok(Dataset::Snips)
        } else {
            Err(Self::Error::Unknown(value.to_string()))
        }
    }
}

impl Display for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Dataset::Snips => snips::DATASET,
        };

        write!(f, "{}", name)
    }
}

/// Dataset Error
#[derive(thiserror::Error, Debug)]
pub enum DatasetError {
    /// No model found for the given string
    #[error("no dataset found for {0}")]
    Unknown(String),
}
