use std::{fmt::Display, path::PathBuf};

use burn::{
    module::AutodiffModule,
    tensor::{backend::AutodiffBackend, Tensor},
    train::{MultiLabelClassificationOutput, TrainStep},
};

use crate::pipelines::sequence_classification;

use super::batcher::{self, Train};

/// A trait for models that can be used for Text Classification
pub trait Model<B>:
    AutodiffModule<B> + TrainStep<Train<B>, MultiLabelClassificationOutput<B>> + Display
where
    B: AutodiffBackend,
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    /// The model configuration
    type Config: ModelConfig;

    /// Perform a forward pass
    fn forward(&self, item: batcher::Train<B>) -> MultiLabelClassificationOutput<B>;

    /// Defines forward pass for inference
    fn infer(&self, input: sequence_classification::batcher::Infer<B>) -> Tensor<B, 2>;

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
        labels: &[String],
    ) -> impl std::future::Future<Output = anyhow::Result<Self>> + Send;

    /// Return the Config needed for the text classification pipeline
    fn get_config(&self) -> sequence_classification::Config;
}
