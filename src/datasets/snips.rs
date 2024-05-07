use burn::data::dataset::{self, Dataset as _, InMemDataset};
use derive_new::new;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::pipelines::text_classification;

/// The name of the Snips dataset
pub static DATASET: &str = "snips";

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

impl text_classification::Item for Item {
    fn input(&self) -> &str {
        &self.input
    }

    fn class_label(&self) -> &str {
        &self.intent
    }
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
    pub async fn load(data_dir: &str, mode: &str) -> std::io::Result<Self> {
        let dataset_dir = format!("{}/datasets/{}", data_dir, DATASET);
        let reader = csv::ReaderBuilder::new();

        let dataset: InMemDataset<Item> =
            InMemDataset::from_csv(format!("{}/{}.csv", dataset_dir, mode), &reader)?;

        Ok(Self { dataset })
    }

    /// Returns random samples from the dataset
    pub async fn get_samples(data_dir: &str) -> std::io::Result<Vec<(String, String)>> {
        let mut rng = rand::thread_rng();

        let data = Self::load(data_dir, "train").await?;

        let mut samples = Vec::with_capacity(10);
        for _ in 0..10 {
            let i = rng.gen_range(0..data.len());
            let item = data.get(i).unwrap();

            samples.push((item.input, item.intent));
        }

        Ok(samples)
    }
}
