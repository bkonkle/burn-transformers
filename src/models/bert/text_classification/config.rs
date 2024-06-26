//! Adapt Bert for Sequence Classification to the Text Classification pipeline

use std::{collections::BTreeMap, path::PathBuf};

use bert_burn::model::BertModelConfig;
use burn::{
    config::Config as _,
    nn::LinearConfig,
    tensor::backend::{AutodiffBackend, Backend},
};
use tokio::io;

use crate::pipelines::sequence_classification::{self, text_classification};

use super::Model;

/// The Model Configuration
#[derive(burn::config::Config)]
pub struct Config {
    /// The base BERT config
    pub model: BertModelConfig,

    /// A map from class ids to class name labels
    pub id2label: BTreeMap<usize, String>,
}

impl Config {
    /// Load configuration from file for training a particular task
    pub fn new_with_labels(model: BertModelConfig, labels: &[String]) -> io::Result<Self> {
        let id2label = labels
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.trim().to_string()))
            .collect();

        Ok(Config::new(model, id2label))
    }

    /// Initialize the model
    pub fn init<B: AutodiffBackend>(&self, device: &B::Device) -> Model<B> {
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

impl text_classification::ModelConfig for Config {
    /// Initialize the model
    fn init<B: AutodiffBackend>(&self, device: &B::Device) -> impl text_classification::Model<B>
    where
        i64: From<<B as Backend>::IntElem>,
    {
        self.init(device)
    }

    /// Load a pretrained model configuration
    async fn load_pretrained(config_file: PathBuf, labels: &[String]) -> anyhow::Result<Self> {
        let mut bert_config = BertModelConfig::load(config_file)
            .map_err(|e| anyhow!("Unable to load Hugging Face Config file: {}", e))?;

        // Enable the pooling layer for sequence classification
        bert_config.with_pooling_layer = Some(true);

        let model_config = Config::new_with_labels(bert_config, labels)?;

        let n_classes = model_config.id2label.len();
        if n_classes == 0 {
            return Err(anyhow::anyhow!(
                "Classes are not defined in the model configuration"
            ));
        }

        Ok(model_config)
    }

    fn get_config(&self) -> sequence_classification::Config {
        sequence_classification::Config {
            pad_token_id: self.model.pad_token_id,
            max_position_embeddings: self.model.max_position_embeddings,
            hidden_size: self.model.hidden_size,
            max_seq_len: self.model.max_seq_len,
            hidden_dropout_prob: self.model.hidden_dropout_prob,
            id2label: self.id2label.clone(),
        }
    }
}
