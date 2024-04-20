use std::collections::HashMap;

use burn::{tensor::backend::Backend, train::ClassificationOutput};

use super::batcher;

/// Text Classification Pipeline
pub struct Pipeline {}

/// A trait for models that can be used for Text Classification
pub trait Model<B: Backend> {
    fn forward(&self, item: batcher::Train<B>) -> ClassificationOutput<B>;
}

/// A trait for configs that can be used for Text Classification models
pub trait ModelConfig: burn::config::Config {
    /// Initialize the model
    fn init<B: Backend>(&self, device: &B::Device) -> impl Model<B>;

    /// Return a mapping from class ids to class names
    fn id2label(&self) -> HashMap<usize, String>;
}

/// A trait for configs that can be used for the pre-trained base models in a Text
/// Classification pipeline
pub trait PreTrainedModelConfig: burn::config::Config {
    fn to_model_config(&self) -> impl ModelConfig;

    /// Return the embedding size (e.g., 768 for roberta-base)
    fn hidden_size(&self) -> usize;

    /// Return the maximum sequence length, if available
    fn max_seq_len(&self) -> Option<usize>;

    /// Return the hidden dropout probability
    fn hidden_dropout_prob(&self) -> f64;

    /// Set the maximum sequence length
    fn set_max_seq_len(&mut self, max_seq_len: Option<usize>);

    /// Set the hidden dropout probability
    fn set_hidden_dropout_prod(&mut self, hidden_dropout_prob: f64);
}
