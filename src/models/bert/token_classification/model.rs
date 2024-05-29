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
};
use derive_new::new;

use crate::{
    models::bert::token_classification::Config,
    pipelines::sequence_classification::{
        self,
        token_classification::{self, Output},
    },
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
    pub fn forward(&self, input: BertInferenceBatch<B>, targets: Tensor<B, 2, Int>) -> Output<B>
    where
        i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
    {
        let [batch_size, seq_length] = input.tokens.dims();
        let device = &self.model.devices()[0];

        let targets = targets.to_device(device);

        let BertModelOutput { hidden_states, .. } = self.model.forward(input);

        let output = self
            .output
            .forward(hidden_states)
            .slice([0..batch_size, 0..seq_length])
            .reshape([batch_size, seq_length, self.n_classes]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.model.embeddings.pad_token_idx]))
            .init(&output.device())
            .forward(
                output
                    .clone()
                    .to_device(device)
                    .reshape([batch_size * seq_length, self.n_classes]),
                targets.clone().reshape([batch_size * seq_length]),
            );

        Output {
            loss,
            output,
            targets,
        }
    }

    /// Defines forward pass for inference
    pub fn infer(&self, input: BertInferenceBatch<B>) -> Tensor<B, 3> {
        let [batch_size, seq_length] = input.tokens.dims();

        let BertModelOutput { hidden_states, .. } = self.model.forward(input);

        let output = self
            .output
            .forward(hidden_states)
            .slice([0..batch_size, 0..seq_length])
            .reshape([batch_size, seq_length, self.n_classes]);

        softmax(output, 2)
    }
}

impl<B: AutodiffBackend> token_classification::Model<B> for Model<B>
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

        let record = ModelRecord {
            model: BertModel::from_safetensors(model_file, device, config.model.clone()),
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
    fn forward(&self, item: token_classification::batcher::Train<B>) -> Output<B> {
        self.forward(
            BertInferenceBatch {
                tokens: item.input.tokens,
                mask_pad: item.input.mask_pad,
            },
            item.targets,
        )
    }

    /// Defines forward pass for inference
    fn infer(&self, input: sequence_classification::batcher::Infer<B>) -> Tensor<B, 3> {
        self.infer(BertInferenceBatch {
            tokens: input.tokens,
            mask_pad: input.mask_pad,
        })
    }
}
