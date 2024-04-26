//! Adapt Bert for Sequence Classification to the Text Classification pipeline

use std::path::PathBuf;

use bert_burn::{
    data::BertInferenceBatch,
    model::{BertModel, BertModelConfig},
};
use burn::{
    config::Config as _,
    module::{ConstantRecord, Module},
    nn::{LinearConfig, LinearRecord},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::pipelines::text_classification::{self, batcher};

use super::{Config, Model, ModelRecord};

/// Available models to use with Bert for Text Classification
pub static MODELS: &[&str] = &["bert-base-cased", "bert-base-uncased"];

/// The default model to use
pub static DEFAULT_MODEL: &str = "bert-base-uncased";

/// Define training step
impl<B: AutodiffBackend> TrainStep<batcher::Train<B>, ClassificationOutput<B>> for Model<B>
where
    i64: From<<B as Backend>::IntElem>,
{
    fn step(&self, item: batcher::Train<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Run forward pass, calculate gradients and return them along with the output
        let output = self.forward(
            BertInferenceBatch {
                tokens: item.input.tokens,
                mask_pad: item.input.mask_pad,
            },
            item.targets,
        );
        let grads = output.loss.backward();

        TrainOutput::new(self, grads, output)
    }
}

/// Define validation step
impl<B: Backend> ValidStep<batcher::Train<B>, ClassificationOutput<B>> for Model<B>
where
    i64: From<<B as Backend>::IntElem>,
{
    fn step(&self, item: batcher::Train<B>) -> ClassificationOutput<B> {
        // Run forward pass and return the output
        self.forward(
            BertInferenceBatch {
                tokens: item.input.tokens,
                mask_pad: item.input.mask_pad,
            },
            item.targets,
        )
    }
}

impl<B: AutodiffBackend> text_classification::Model<B> for Model<B>
where
    i64: From<<B as Backend>::IntElem>,
{
    /// The model configuration
    type Config = Config;

    /// Load a model from a file
    fn load_from_safetensors(
        device: &B::Device,
        model_file: PathBuf,
        config: Self::Config,
    ) -> anyhow::Result<Self> {
        let n_classes = config.id2label.len();
        if n_classes == 0 {
            return Err(anyhow::anyhow!(
                "Classes are not defined in the model configuration"
            ));
        }

        // Initialize the linear output
        let output = LinearConfig::new(config.model.hidden_size, n_classes).init(device);

        let model = config.init(device).load_record(ModelRecord {
            model: BertModel::from_safetensors(model_file, device, config.model, true),
            output: LinearRecord {
                weight: output.weight,
                bias: output.bias,
            },
            n_classes: ConstantRecord::new(),
        });

        Ok(model)
    }

    /// Perform a forward pass
    fn forward(&self, item: batcher::Train<B>) -> ClassificationOutput<B> {
        self.forward(
            BertInferenceBatch {
                tokens: item.input.tokens,
                mask_pad: item.input.mask_pad,
            },
            item.targets,
        )
    }

    /// Defines forward pass for inference
    fn infer(&self, input: batcher::Infer<B>) -> Tensor<B, 2> {
        self.infer(BertInferenceBatch {
            tokens: input.tokens,
            mask_pad: input.mask_pad,
        })
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
    async fn load_pretrained(config_file: PathBuf, dataset_dir: &str) -> anyhow::Result<Self> {
        let bert_config = BertModelConfig::load(config_file)
            .map_err(|e| anyhow!("Unable to load Hugging Face Config file: {}", e))?;

        let model_config = Config::new_for_task(bert_config, dataset_dir).await?;

        let n_classes = model_config.id2label.len();
        if n_classes == 0 {
            return Err(anyhow::anyhow!(
                "Classes are not defined in the model configuration"
            ));
        }

        Ok(model_config)
    }

    fn get_config(&self) -> text_classification::Config {
        text_classification::Config {
            pad_token_id: self.model.pad_token_id,
            max_position_embeddings: self.model.max_position_embeddings,
            hidden_size: self.model.hidden_size,
            max_seq_len: self.model.max_seq_len,
            hidden_dropout_prob: self.model.hidden_dropout_prob,
            id2label: self.id2label.clone(),
            label2id: self.label2id.clone(),
        }
    }
}
