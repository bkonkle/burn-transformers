use std::collections::HashMap;

use bert_burn::model::BertModelConfig;
use burn::{nn::LinearConfig, tensor::backend::Backend};
use tokio::io;

use super::model::Model;

/// The Model Configuration
#[derive(burn::config::Config)]
pub struct Config {
    /// The base BERT config
    pub model: BertModelConfig,

    /// A map from class ids to class name labels
    pub id2label: HashMap<usize, String>,

    /// A reverse map from class name labels to class ids
    pub label2id: HashMap<String, usize>,
}

impl Config {
    /// Load configuration from file for training a particular task
    pub fn new_with_labels(model: BertModelConfig, labels: &[String]) -> io::Result<Self> {
        let id2label = labels
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.trim().to_string()))
            .collect::<HashMap<_, _>>();

        let label2id = id2label
            .iter()
            .map(|(k, v)| (v.clone(), *k))
            .collect::<HashMap<_, _>>();

        Ok(Config::new(model, id2label, label2id))
    }

    /// Initializes a Bert model with default weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let model = self.model.init(device);

        let n_classes = self.id2label.len();

        let output = LinearConfig::new(self.model.hidden_size, n_classes).init(device);

        Model {
            model,
            output,
            n_classes,
        }
    }
}
