use burn::{
    tensor::{backend::Backend, Int, Tensor},
    train::metric::{Adaptor, LossInput},
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
}

impl<B: Backend> Adaptor<LossInput<B>> for Output<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}
