use std::path::PathBuf;

use bert_burn::{
    data::BertInferenceBatch,
    model::{BertModel, BertModelOutput},
};
use burn::{
    module::{ConstantRecord, Module},
    nn::{loss::CrossEntropyLossConfig, Linear, LinearConfig, LinearRecord},
    tensor::{
        activation::softmax,
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
    train::ClassificationOutput,
};
use derive_new::new;

use crate::{
    models::bert::text_classification::Config,
    pipelines::sequence_classification::{self, text_classification},
};

/// BERT for sequence Classification
#[derive(Module, Debug, new)]
pub struct Model<B: Backend> {
    /// The base BERT model
    pub model: BertModel<B>,

    /// Linear layer for sequence classification
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
        let output = LinearConfig::new(config.hidden_size, n_classes).init(device);

        let record = ModelRecord {
            model: BertModel::from_safetensors(model_file, device, config.get_bert_config()),
            output: LinearRecord {
                weight: output.weight,
                bias: output.bias,
            },
            n_classes: ConstantRecord::new(),
        };

        let model = config.init(device).load_record(record);

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
    fn infer(&self, input: sequence_classification::batcher::Infer<B>) -> Tensor<B, 2> {
        self.infer(BertInferenceBatch {
            tokens: input.tokens,
            mask_pad: input.mask_pad,
        })
    }
}
