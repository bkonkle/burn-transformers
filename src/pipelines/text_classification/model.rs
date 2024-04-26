use std::{collections::HashMap, fmt::Display, path::PathBuf};

use burn::{
    module::AutodiffModule,
    tensor::{backend::AutodiffBackend, Tensor},
    train::{ClassificationOutput, TrainStep},
};

use super::batcher::{self, Train};

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

/// A trait for models that can be used for Text Classification
pub trait Model<B>:
    AutodiffModule<B> + TrainStep<Train<B>, ClassificationOutput<B>> + Display
where
    B: AutodiffBackend,
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    /// The model configuration
    type Config: ModelConfig;

    /// Perform a forward pass
    fn forward(&self, item: batcher::Train<B>) -> ClassificationOutput<B>;

    /// Defines forward pass for inference
    fn infer(&self, input: batcher::Infer<B>) -> Tensor<B, 2>;

    /// Load a model from a file
    fn load_from_safetensors(
        device: &B::Device,
        model_file: PathBuf,
        config: Self::Config,
    ) -> anyhow::Result<Self>;
}

/// A trait for configs that can be used for Text Classification models
pub trait ModelConfig: burn::config::Config + Clone {
    /// Initialize the model
    fn init<B: AutodiffBackend>(&self, device: &B::Device) -> impl Model<B>
    where
        i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>;

    /// Load a pretrained model configuration
    fn load_pretrained(
        config_file: PathBuf,
        dataset_dir: &str,
    ) -> impl std::future::Future<Output = anyhow::Result<Self>> + Send;

    /// Return the Config needed for the text classification pipeline
    fn get_config(&self) -> Config;
}
