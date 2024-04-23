use bert_burn::{
    data::BertInferenceBatch,
    model::{BertModel, BertModelOutput},
};
use burn::{
    module::Module,
    nn::{loss::CrossEntropyLossConfig, Linear},
    tensor::{activation::softmax, backend::Backend, Int, Tensor},
    train::ClassificationOutput,
};
use derive_new::new;

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
