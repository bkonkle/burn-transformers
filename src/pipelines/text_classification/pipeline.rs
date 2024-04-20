use std::{collections::HashMap, fmt::Display, path::PathBuf};

use burn::{
    module::AutodiffModule,
    tensor::{backend::AutodiffBackend, Tensor},
    train::{ClassificationOutput, TrainStep},
};

use super::batcher::{self, Train};

/// Text Classification Pipeline
pub struct Pipeline {}

/// A trait for models that can be used for Text Classification
pub trait Model<B>:
    AutodiffModule<B> + TrainStep<Train<B>, ClassificationOutput<B>> + Display
where
    B: AutodiffBackend,
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    /// The model configuration
    type Config: ModelConfig;

    /// Load a model from a file
    fn load_from_safetensors(
        device: &B::Device,
        model_file: PathBuf,
        model_config: Self::Config,
    ) -> Self;

    /// Perform a forward pass
    fn forward(&self, item: batcher::Train<B>) -> ClassificationOutput<B>;

    /// Defines forward pass for inference
    fn infer(&self, input: batcher::Infer<B>) -> Tensor<B, 2>;
}

/// A trait for configs that can be used for Text Classification models
pub trait ModelConfig: burn::config::Config + Clone {
    /// Load a pretrained model configuration
    fn load_pretrained(
        config_file: PathBuf,
        max_seq_len: usize,
        hidden_dropout_prob: f64,
    ) -> anyhow::Result<Self>;

    /// Initialize the model
    fn init<B: AutodiffBackend, M: Model<B>>(&self, device: &B::Device) -> M
    where
        i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>;

    /// Return the padding token ID
    fn pad_token_id(&self) -> usize;

    /// Return the maximum sequence length
    fn max_position_embeddings(&self) -> usize;

    /// Return the embedding size (e.g., 768 for roberta-base)
    fn hidden_size(&self) -> usize;

    /// Return the maximum sequence length, if available
    fn max_seq_len(&self) -> Option<usize>;

    /// Return the hidden dropout probability
    fn hidden_dropout_prob(&self) -> f64;

    /// Set the hidden dropout probability
    fn set_hidden_dropout_prob(&mut self, value: f64);

    /// Return a mapping from class ids to class name labels
    fn id2label(&self) -> HashMap<usize, String>;

    /// Return a mapping from class name labels to class ids
    fn label2id(&self) -> HashMap<String, usize>;
}
