#![allow(clippy::too_many_arguments)]

use std::{collections::HashMap, path::PathBuf};

use bert_burn::{
    data::BertInferenceBatch,
    model::{BertModel, BertModelConfig, BertModelOutput},
};
use burn::{
    config::Config as _,
    module::{ConstantRecord, Module},
    nn::{loss::CrossEntropyLossConfig, Linear, LinearConfig, LinearRecord},
    tensor::{
        activation::softmax,
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use derive_new::new;
use tokio::io;

use crate::{pipelines::text_classification, utils::files::read_file};

/// BERT for text Classification
#[derive(Module, Debug, new)]
pub struct Model<B: Backend> {
    /// The base BERT model
    pub model: BertModel<B>,

    /// Linear layer for text classification
    pub output: Linear<B>,

    /// Total number of classes
    pub n_classes: usize,
}

/// Define model behavior
impl<B: Backend> Model<B> {
    /// Defines forward pass for training
    pub fn forward(
        &self,
        input: BertInferenceBatch<B>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B>
    where
        i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
    {
        let [batch_size, _seq_length] = input.tokens.dims();
        let device = &self.model.devices()[0];

        let targets = targets.to_device(device);

        let BertModelOutput {
            pooled_output,
            hidden_states,
        } = self.model.forward(input);

        let output = self
            .output
            .forward(pooled_output.unwrap_or(hidden_states))
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }

    /// Defines forward pass for inference
    pub fn infer(&self, input: BertInferenceBatch<B>) -> Tensor<B, 2> {
        let [batch_size, _seq_length] = input.tokens.dims();

        let BertModelOutput {
            pooled_output,
            hidden_states,
        } = self.model.forward(input);

        let output = self
            .output
            .forward(pooled_output.unwrap_or(hidden_states))
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.n_classes]);

        softmax(output, 1)
    }
}

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
    /// Initializes a Bert model with default weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let model = self.model.init(device, true);

        let n_classes = self.id2label.len();

        let output = LinearConfig::new(self.model.hidden_size, n_classes).init(device);

        Model {
            model,
            output,
            n_classes,
        }
    }

    /// Load configuration from file for training a particular task
    pub async fn new_for_task(model: BertModelConfig, dataset_dir: &str) -> io::Result<Self> {
        let id2label = read_file(&format!("{}/intent_labels.txt", dataset_dir))
            .await?
            .into_iter()
            .enumerate()
            .map(|(i, s)| (i, s.trim().to_string()))
            .collect::<HashMap<_, _>>();

        let label2id = id2label
            .iter()
            .map(|(k, v)| (v.clone(), *k))
            .collect::<HashMap<_, _>>();

        Ok(Config::new(model, id2label, label2id))
    }
}

/// Define training step
impl<B: AutodiffBackend> TrainStep<text_classification::batcher::Train<B>, ClassificationOutput<B>>
    for Model<B>
where
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    fn step(
        &self,
        item: text_classification::batcher::Train<B>,
    ) -> TrainOutput<ClassificationOutput<B>> {
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
impl<B: Backend> ValidStep<text_classification::batcher::Train<B>, ClassificationOutput<B>>
    for Model<B>
where
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    fn step(&self, item: text_classification::batcher::Train<B>) -> ClassificationOutput<B> {
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
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
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
    fn forward(&self, item: text_classification::batcher::Train<B>) -> ClassificationOutput<B> {
        self.forward(
            BertInferenceBatch {
                tokens: item.input.tokens,
                mask_pad: item.input.mask_pad,
            },
            item.targets,
        )
    }

    /// Defines forward pass for inference
    fn infer(&self, input: text_classification::batcher::Infer<B>) -> Tensor<B, 2> {
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
        i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
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
