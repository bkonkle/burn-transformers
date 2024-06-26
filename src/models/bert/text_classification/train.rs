use bert_burn::data::BertInferenceBatch;
use burn::{
    tensor::backend::{AutodiffBackend, Backend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::pipelines::sequence_classification::text_classification::batcher;

use super::Model;

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
