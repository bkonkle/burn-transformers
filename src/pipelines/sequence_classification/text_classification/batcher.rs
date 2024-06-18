#![allow(clippy::too_many_arguments)]

use std::fmt::Debug;

use burn::{
    data::dataloader,
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};
use derive_new::new;
use tokenizers::Tokenizer;

use crate::pipelines::sequence_classification;

use super::Item;

/// A training batch for text classification
#[derive(Clone, Debug, new)]
pub struct Train<B: Backend> {
    /// Bert Model input
    pub input: sequence_classification::batcher::Infer<B>,

    /// Class ids for the batch
    pub targets: Tensor<B, 1, Int>,
}

/// Struct for batching sequence classification items
#[derive(Clone)]
pub struct Batcher<B: Backend> {
    /// Wrap the Text Classification batcher because Token Classification is a variation
    batcher: sequence_classification::Batcher<B>,
}

impl<B: Backend> Batcher<B> {
    /// Creates a new batcher
    pub fn new(
        tokenizer: Tokenizer,
        config: sequence_classification::Config,
        device: B::Device,
    ) -> Self {
        let batcher = sequence_classification::Batcher::new(tokenizer, config, device);

        Self { batcher }
    }
}

/// Implement Batcher trait for Batcher struct for inference
impl<B: Backend> dataloader::batcher::Batcher<String, sequence_classification::batcher::Infer<B>>
    for Batcher<B>
{
    /// Collects a vector of text classification items into a inference batch
    fn batch(&self, items: Vec<String>) -> sequence_classification::batcher::Infer<B> {
        self.batcher.batch(items)
    }
}

/// Implement Batcher trait for Batcher struct for training
impl<B: Backend, I: Item> dataloader::batcher::Batcher<I, Train<B>> for Batcher<B> {
    /// Collects a vector of text classification items into a training batch
    fn batch(&self, items: Vec<I>) -> Train<B> {
        let batch_size = items.len();

        let inputs = items.iter().map(|item| item.input().to_string()).collect();
        let infer: sequence_classification::batcher::Infer<B> = self.batch(inputs);

        let mut class_id_list = Vec::with_capacity(batch_size);

        // Tokenize text and create class_id tensor for each item
        for item in &items {
            let class_id = self
                .batcher
                .label2id
                .get(item.class_label())
                .unwrap_or(&self.batcher.unk_token_id);

            class_id_list.push(Tensor::from_data(
                Data::from([(*class_id as i64).elem()]),
                &self.batcher.device,
            ));
        }

        let targets = Tensor::cat(class_id_list, 0);

        // Create and return training batch
        Train {
            input: infer,
            targets,
        }
    }
}
