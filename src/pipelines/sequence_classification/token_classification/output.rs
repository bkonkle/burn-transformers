use burn::{
    tensor::{backend::Backend, Bool, Int, Tensor},
    train::metric::{AccuracyInput, Adaptor, LossInput},
};
use derive_new::new;

/// Multi-label classification output adapted for multiple metrics.
#[derive(new)]
pub struct Output<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The output.
    pub output: Tensor<B, 3>,

    /// The targets.
    pub targets: Tensor<B, 2, Int>,

    /// The attention mask.
    pub attention_mask: Tensor<B, 2, Bool>,
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for Output<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        let [batch_size, seq_length, n_classes] = self.output.dims();

        AccuracyInput::new(
            self.output
                .clone()
                .reshape([batch_size * seq_length, n_classes]),
            self.targets.clone().reshape([batch_size * seq_length]),
        )
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for Output<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

/// Inference Output without the loss and targets
#[derive(new)]
pub struct Inference<B: Backend> {
    /// The output.
    pub output: Tensor<B, 3>,

    /// The attention mask.
    pub attention_mask: Tensor<B, 2, Bool>,
}
