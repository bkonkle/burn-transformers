use burn::data::dataset::{self, Dataset as _, InMemDataset};
use derive_new::new;
use hf_hub::api::tokio;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    pipelines::sequence_classification::{text_classification, token_classification},
    utils::files::read_file,
};

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

impl token_classification::Item for Item {
    fn input(&self) -> &str {
        &self.input
    }

    fn class_labels(&self) -> Vec<&str> {
        self.slots.split_whitespace().collect()
    }
}

/// Struct for the Snips dataset
pub struct Dataset {
    /// Underlying In-Memory dataset
    dataset: InMemDataset<Item>,

    /// Intent labels
    pub intent_labels: Vec<String>,

    /// Slot labels
    pub slot_labels: Vec<String>,
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
    pub async fn load(mode: &str) -> anyhow::Result<Self> {
        let api = tokio::Api::new().unwrap();
        let repo = api.dataset("bkonkle/snips-joint-intent".to_string());

        let dataset_file = repo.get(&format!("{}.csv", mode)).await?;
        let dataset: InMemDataset<Item> =
            InMemDataset::from_csv(dataset_file, &csv::ReaderBuilder::new())?;

        let intent_labels_file = repo.get("intent_labels.txt").await?;
        let intent_labels = read_file(
            intent_labels_file
                .to_str()
                .expect("unable to read intent_labels.txt path"),
        )
        .await?;

        let slot_labels_file = repo.get("slot_labels.txt").await?;
        let slot_labels = read_file(
            slot_labels_file
                .to_str()
                .expect("unable to read slot_labels.txt path"),
        )
        .await?;

        Ok(Self {
            dataset,
            intent_labels,
            slot_labels,
        })
    }

    /// Returns random samples from the dataset
    pub async fn get_samples(mode: &str) -> anyhow::Result<Vec<(String, String, String)>> {
        let mut rng = rand::thread_rng();

        let data = Self::load(mode).await?;

        let mut samples = Vec::with_capacity(10);
        for _ in 0..10 {
            let i = rng.gen_range(0..data.len());
            let item = data.get(i).unwrap();

            samples.push((item.input, item.intent, item.slots));
        }

        Ok(samples)
    }
}
