#![allow(clippy::too_many_arguments)]

use std::fmt::Debug;

use burn::{
    data::dataloader,
    tensor::{backend::Backend, Int, Tensor},
};
use derive_new::new;
use tokenizers::Tokenizer;

use crate::{pipelines::text_classification, utils::tensors};

use super::Item;

/// A training batch for token classification
#[derive(Clone, Debug, new)]
pub struct Train<B: Backend> {
    /// Bert Model input
    pub input: text_classification::batcher::Infer<B>,

    /// Class ids for the batch
    pub targets: Tensor<B, 2, Int>,
}

/// Struct for batching sequence classification items
#[derive(Clone)]
pub struct Batcher<B: Backend> {
    /// Wrap the Text Classification batcher because Token Classification is a variation
    batcher: text_classification::Batcher<B>,
}

impl<B: Backend> Batcher<B> {
    /// Creates a new batcher
    pub fn new<C: text_classification::ModelConfig>(
        tokenizer: Tokenizer,
        config: C,
        device: B::Device,
    ) -> Self {
        let batcher = text_classification::Batcher::new(tokenizer, config, device);

        Self { batcher }
    }
}

/// Implement Batcher trait for Batcher struct for inference
impl<B: Backend> dataloader::batcher::Batcher<String, text_classification::batcher::Infer<B>>
    for Batcher<B>
{
    /// Collects a vector of text classification items into a inference batch
    fn batch(&self, items: Vec<String>) -> text_classification::batcher::Infer<B> {
        self.batcher.batch(items)
    }
}

/// Implement Batcher trait for Batcher struct for training
impl<B: Backend, I: Item> dataloader::batcher::Batcher<I, Train<B>> for Batcher<B> {
    /// Collects a vector of text classification items into a training batch
    fn batch(&self, items: Vec<I>) -> Train<B> {
        let batch_size = items.len();

        let inputs = items.iter().map(|item| item.input().to_string()).collect();
        let infer: text_classification::batcher::Infer<B> = self.batch(inputs);

        let seq_length = infer.tokens.dims()[1];

        let mut class_ids_list = Vec::with_capacity(batch_size);

        // Tokenize text and create class_id tensor for each item
        for item in items {
            let class_ids: Vec<_> = item
                .class_labels()
                .iter()
                .map(|slot| self.batcher.reverse_map.get(*slot).unwrap_or(&0).to_owned())
                .collect();

            class_ids_list.push(class_ids);
        }

        // Pad the slot labels to match the tokenized sequence length
        let targets = tensors::pad_to::<B>(
            self.batcher.pad_token_id,
            class_ids_list,
            seq_length,
            &self.batcher.device,
        );

        // Create and return training batch
        Train {
            input: infer,
            targets,
        }
    }
}
