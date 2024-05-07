use crate::datasets::snips;

/// The Dataset enum
pub enum Dataset {
    /// Snips dataset
    Snips,
}

impl TryFrom<String> for Dataset {
    type Error = DatasetError;

    /// Try to convert a string to a Dataset
    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value == *snips::DATASET {
            Ok(Dataset::Snips)
        } else {
            Err(Self::Error::Unknown(value))
        }
    }
}

impl From<Dataset> for String {
    fn from(dataset: Dataset) -> Self {
        match dataset {
            Dataset::Snips => snips::DATASET.to_string(),
        }
    }
}

/// Dataset Error
#[derive(thiserror::Error, Debug)]
pub enum DatasetError {
    /// No model found for the given string
    #[error("no dataset found for {0}")]
    Unknown(String),
}
