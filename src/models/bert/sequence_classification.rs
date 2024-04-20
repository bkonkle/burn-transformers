#![allow(clippy::too_many_arguments)]

use std::{collections::HashMap, path::PathBuf};

use bert_burn::{
    data::BertInferenceBatch,
    model::{BertModel, BertModelConfig, BertModelOutput},
};
use burn::{
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

use crate::pipelines::text_classification;

/// The Bert model name on Hugging Face
pub const MODEL_NAME: &str = "bert-base-uncased";

/// A training batch for text classification
#[derive(Clone, Debug, new)]
pub struct Train<B: Backend> {
    /// Bert Model input
    pub input: BertInferenceBatch<B>,

    /// Class ids for the batch
    pub targets: Tensor<B, 1, Int>,
}

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
    pub fn forward(&self, item: Train<B>) -> ClassificationOutput<B>
    where
        i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
    {
        let [batch_size, _seq_length] = item.input.tokens.dims();
        let device = &self.model.devices()[0];

        let targets = item.targets.to_device(device);

        let BertModelOutput {
            pooled_output,
            hidden_states,
        } = self.model.forward(item.input);

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

    /// A map from class ids to class names
    pub id2label: HashMap<usize, String>,
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

    /// Generate a map from class names to class ids
    pub fn get_reverse_class_map(&self) -> HashMap<String, usize> {
        let mut swapped_map: HashMap<String, usize> = HashMap::new();

        // Iterate through the original map
        for (key, value) in &self.id2label {
            // Insert the value as a key and the key as a value into the new map
            swapped_map.insert(value.clone(), *key);
        }

        swapped_map
    }
}

/// Define training step
impl<B: AutodiffBackend> TrainStep<Train<B>, ClassificationOutput<B>> for Model<B>
where
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    fn step(&self, item: Train<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Run forward pass, calculate gradients and return them along with the output
        let output = self.forward(item);
        let grads = output.loss.backward();

        TrainOutput::new(self, grads, output)
    }
}

/// Define validation step
impl<B: Backend> ValidStep<Train<B>, ClassificationOutput<B>> for Model<B>
where
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    fn step(&self, item: Train<B>) -> ClassificationOutput<B> {
        // Run forward pass and return the output
        self.forward(item)
    }
}

impl<B: AutodiffBackend> text_classification::pipeline::Model<B> for Model<B>
where
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    /// The model configuration
    type Config = Config;

    /// Load a model from a file
    fn load_from_safetensors(
        device: &B::Device,
        model_file: PathBuf,
        model_config: Self::Config,
    ) -> Self {
        BertModel::from_safetensors(model_file, device, model_config, true)
    }

    /// Perform a forward pass
    fn forward(&self, item: text_classification::batcher::Train<B>) -> ClassificationOutput<B> {
        self.forward(item)
    }

    /// Defines forward pass for inference
    fn infer(&self, input: text_classification::batcher::Infer<B>) -> Tensor<B, 2> {
        self.infer(input)
    }
}

impl<B: AutodiffBackend> text_classification::pipeline::ModelConfig for Model<B> {
    /// Load a pretrained model configuration
    fn load_pretrained(
        config_file: PathBuf,
        max_seq_len: usize,
        hidden_dropout_prob: f64,
    ) -> anyhow::Result<Self> {
        let mut bert_config = BertModelConfig::load(config_file)
            .map_err(|e| anyhow!("Unable to load Hugging Face Config file: {}", e))?;

        bert_config.max_seq_len = Some(max_seq_len);
        bert_config.hidden_dropout_prob = hidden_dropout_prob;

        let model_config = Config::new(bert_config.clone(), config.id2label);

        let n_classes = model_config.id2label.len();

        if n_classes == 0 {
            return Err(anyhow::anyhow!(
                "Classes are not defined in the model configuration"
            ));
        }

        // Initialize the linear output
        let output = LinearConfig::new(model_config.model.hidden_size, n_classes).init(device);

        model_config.init(device).load_record(ModelRecord {
            model: BertModel::from_safetensors(model_file, device, bert_config, true),
            output: LinearRecord {
                weight: output.weight,
                bias: output.bias,
            },
            n_classes: ConstantRecord::new(),
        })
    }

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
