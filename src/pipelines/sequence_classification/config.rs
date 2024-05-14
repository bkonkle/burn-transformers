use std::collections::HashMap;

use burn::LearningRate;

/// The common model configuration properties needed for the pipeline
pub struct Config {
    /// The padding token ID
    pub pad_token_id: usize,

    /// The max position embeddings
    pub max_position_embeddings: usize,

    /// The size of the hidden state
    pub hidden_size: usize,

    /// An optional max sequence length, if different from max position embeddings
    pub max_seq_len: Option<usize>,

    /// The hidden dropout probability
    pub hidden_dropout_prob: f64,

    /// A mapping from class ids to class name labels
    pub id2label: HashMap<usize, String>,

    /// A mapping from class name labels to class ids
    pub label2id: HashMap<String, usize>,
}

/// Define configuration struct for the experiment
#[derive(burn::config::Config)]
pub struct Training {
    /// Batch size
    #[config(default = 2)]
    pub batch_size: usize,

    /// Number of epochs
    #[config(default = 1)]
    pub num_epochs: usize,

    /// Adam epsilon
    #[config(default = 1e-8)]
    pub adam_epsilon: f32,

    /// Initial learning rate
    #[config(default = 1e-2)]
    pub learning_rate: LearningRate,

    /// Dropout rate
    #[config(default = 0.1)]
    pub hidden_dropout_prob: f64,

    /// The location of the top-level data directory
    #[config(default = "\"data\".to_string()")]
    pub data_dir: String,

    /// Model name (e.g., "bert-base-uncased")
    pub model_name: String,

    /// The Dataset to use (e.g., "snips")
    pub dataset_name: String,

    /// Class labels for the selected dataset
    pub labels: Vec<String>,
}
