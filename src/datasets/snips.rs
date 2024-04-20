use burn::data::dataset::{self, InMemDataset};
use derive_new::new;
use serde::{Deserialize, Serialize};

/// Define a struct for Snips text and token classification items
#[derive(Clone, Debug, Serialize, Deserialize, new)]
pub struct Item {
    /// The text for classification
    pub input: String,

    /// The intent class name of the text
    pub intent: String,

    /// The slot class names of the text
    pub slots: String,
}

/// Struct for the Snips dataset
pub struct Dataset {
    /// Underlying In-Memory dataset
    dataset: InMemDataset<Item>,
}

/// Implement the Dataset trait for the Snips dataset
impl dataset::Dataset<Item> for Dataset {
    /// Returns a specific item from the dataset
    fn get(&self, index: usize) -> Option<Item> {
        self.dataset.get(index)
    }

    /// Returns the length of the dataset
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

// Implement methods for constructing the Snips dataset
impl Dataset {
    /// Constructs the dataset for a mode (either "train" or "test")
    pub async fn load(data_root: &str, mode: &str) -> std::io::Result<Self> {
        let task_root = format!("{}/snips", data_root);
        let reader = csv::ReaderBuilder::new();

        let dataset: InMemDataset<Item> =
            InMemDataset::from_csv(format!("{}/{}.csv", task_root, mode), &reader)?;

        Ok(Self { dataset })
    }

    /// Returns the training portion of the dataset
    pub async fn train() -> std::io::Result<Self> {
        Self::load("data", "train").await
    }

    /// Returns the testing portion of the dataset
    pub async fn test() -> std::io::Result<Self> {
        Self::load("data", "test").await
    }
}
