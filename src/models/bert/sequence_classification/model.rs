#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use bert_burn::{
    data::BertInferenceBatch,
    model::{BertModel, BertModelConfig, BertModelOutput},
};
use burn::{
    module::Module,
    nn::{loss::CrossEntropyLossConfig, Linear, LinearConfig},
    tensor::{
        activation::softmax,
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use derive_new::new;

use super::batcher;

/// The Bert model name on Hugging Face
pub const MODEL_NAME: &str = "bert-base-uncased";

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
    pub fn forward(&self, item: batcher::Train<B>) -> ClassificationOutput<B>
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

/// Model input
#[derive(new)]
pub struct Input<B: Backend> {
    /// Outputs
    pub outputs: Tensor<B, 2>,

    /// Targets
    pub targets: Tensor<B, 1, Int>,
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> super::Model<B> {
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
impl<B: AutodiffBackend> TrainStep<batcher::Train<B>, ClassificationOutput<B>> for Model<B>
where
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    fn step(&self, item: batcher::Train<B>) -> TrainOutput<ClassificationOutput<B>> {
        // Run forward pass, calculate gradients and return them along with the output
        let output = self.forward(item);
        let grads = output.loss.backward();

        TrainOutput::new(self, grads, output)
    }
}

/// Define validation step
impl<B: Backend> ValidStep<batcher::Train<B>, ClassificationOutput<B>> for Model<B>
where
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    fn step(&self, item: batcher::Train<B>) -> ClassificationOutput<B> {
        // Run forward pass and return the output
        self.forward(item)
    }
}
