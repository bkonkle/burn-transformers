//! Adapt Bert for Sequence Classification to the Token Classification pipeline

use bert_burn::data::BertInferenceBatch;
use burn::{
    tensor::backend::{AutodiffBackend, Backend},
    train::{MultiLabelClassificationOutput, TrainOutput, TrainStep},
};

use crate::pipelines::sequence_classification::token_classification::batcher;

/// Define training step
impl<B: AutodiffBackend> TrainStep<batcher::Train<B>, MultiLabelClassificationOutput<B>>
    for Model<B>
where
    i64: From<<B as Backend>::IntElem>,
{
    fn step(&self, item: batcher::Train<B>) -> TrainOutput<MultiLabelClassificationOutput<B>> {
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
