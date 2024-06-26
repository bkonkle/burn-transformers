#![allow(clippy::too_many_arguments)]

use std::fmt::Debug;

use burn::{
    data::dataloader,
    tensor::{backend::Backend, Int, Tensor},
};
use derive_new::new;
use tokenizers::Tokenizer;

use crate::{pipelines::sequence_classification, utils::tensors};

use super::Item;

/// A training batch for token classification
#[derive(Clone, Debug, new)]
pub struct Train<B: Backend> {
    /// Bert Model input
    pub input: sequence_classification::batcher::Infer<B>,

    /// Class ids for the batch
    pub targets: Tensor<B, 2, Int>,
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

        let seq_length = infer.tokens.dims()[1];

        let mut class_ids_list = Vec::with_capacity(batch_size);

        let attention_masks = infer.attention_mask.to_data().value;
        let attention_masks = attention_masks.chunks(seq_length).collect::<Vec<_>>();

        // Tokenize text and create class_id tensor for each item
        for (i, item) in items.iter().enumerate() {
            let mut class_ids: Vec<_> = item
                .class_labels()
                .iter()
                .map(|slot| self.batcher.label2id.get(*slot).unwrap_or(&0).to_owned())
                .collect();

            // Insert padding to account for special tokens and wordpieces, which the dataset won't
            // include
            for (j, attention) in attention_masks[i].iter().enumerate() {
                if !*attention {
                    let class_ids_len = class_ids.len();

                    if j < class_ids_len {
                        class_ids.insert(j, self.batcher.pad_token_id);
                    } else if class_ids_len < seq_length {
                        class_ids.push(self.batcher.pad_token_id);
                    }
                }
            }

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
