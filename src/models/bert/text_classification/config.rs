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
    // -- Fields copied from BertModelConfig because #[serde(flatten)] is not supported yet
    /// Number of attention heads in the multi-head attention
    pub num_attention_heads: usize,
    /// Number of transformer encoder layers/blocks
    pub num_hidden_layers: usize,
    /// Layer normalization epsilon
    pub layer_norm_eps: f64,
    /// Size of bert embedding (e.g., 768 for roberta-base)
    pub hidden_size: usize,
    /// Size of the intermediate position wise feedforward layer
    pub intermediate_size: usize,
    /// Size of the vocabulary
    pub vocab_size: usize,
    /// Max position embeddings, in RoBERTa equal to max_seq_len + 2 (514), for BERT equal to max_seq_len(512)
    pub max_position_embeddings: usize,
    /// Identifier for sentence type in input (e.g., 0 for single sentence, 1 for pair)
    pub type_vocab_size: usize,
    /// Dropout value across layers, typically 0.1
    pub hidden_dropout_prob: f64,
    /// BERT model name (roberta)
    pub model_type: String,
    /// Index of the padding token
    pub pad_token_id: usize,
    /// Maximum sequence length for the tokenizer
    pub max_seq_len: Option<usize>,
    /// Whether to add a pooling layer to the model
    pub with_pooling_layer: Option<bool>,
    // -- End fields copied from BertModelConfig
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

        let config = Config::new(
            model.num_attention_heads,
            model.num_hidden_layers,
            model.layer_norm_eps,
            model.hidden_size,
            model.intermediate_size,
            model.vocab_size,
            model.max_position_embeddings,
            model.type_vocab_size,
            model.hidden_dropout_prob,
            model.model_type,
            model.pad_token_id,
            id2label,
        )
        .with_max_seq_len(model.max_seq_len)
        .with_with_pooling_layer(model.with_pooling_layer);

        Ok(config)
    }

    /// Get the Bert model configuration
    pub fn get_bert_config(&self) -> BertModelConfig {
        BertModelConfig::new(
            self.num_attention_heads,
            self.num_hidden_layers,
            self.layer_norm_eps,
            self.hidden_size,
            self.intermediate_size,
            self.vocab_size,
            self.max_position_embeddings,
            self.type_vocab_size,
            self.hidden_dropout_prob,
            self.model_type.clone(),
            self.pad_token_id,
        )
        .with_max_seq_len(self.max_seq_len)
        .with_with_pooling_layer(self.with_pooling_layer)
    }

    /// Initialize the model
    pub fn init<B: AutodiffBackend>(&self, device: &B::Device) -> Model<B> {
        let model = self.get_bert_config().init(device);

        let n_classes = self.id2label.len();

        let output = LinearConfig::new(self.hidden_size, n_classes).init(device);

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
            pad_token_id: self.pad_token_id,
            max_position_embeddings: self.max_position_embeddings,
            hidden_size: self.hidden_size,
            max_seq_len: self.max_seq_len,
            hidden_dropout_prob: self.hidden_dropout_prob,
            id2label: self.id2label.clone(),
        }
    }
}
