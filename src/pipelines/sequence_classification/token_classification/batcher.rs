#![allow(clippy::too_many_arguments)]

use std::fmt::Debug;

use burn::{
    data::dataloader,
    nn::attention::generate_padding_mask,
    tensor::{backend::Backend, Int, Tensor},
};
use derive_new::new;
use tokenizers::Tokenizer;

use crate::{
    pipelines::sequence_classification::{self, batcher::Infer},
    utils::tensors,
};

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
        // We can't re-use the "Infer" workflow here, because we need to pad the slot labels for
        // words that are split into multiple tokens when tokenized
        let batch_size = items.len();

        let mut token_ids_list = Vec::with_capacity(batch_size);
        let mut class_ids_list = Vec::with_capacity(batch_size);

        // Tokenize text and create class_id tensor for each item
        for item in items {
            // Tokens
            let tokens = self
                .batcher
                .tokenizer
                .encode(item.input(), true)
                .expect("unable to encode");

            let token_ids: Vec<_> = tokens.get_ids().iter().map(|t| *t as usize).collect();

            token_ids_list.push(token_ids);

            // Target class ids
            let mut class_ids: Vec<_> = item
                .class_labels()
                .iter()
                .map(|slot| self.batcher.reverse_map.get(*slot).unwrap_or(&0).to_owned())
                .collect();

            // Keep a separate index for the class labels, skipping special tokens
            let mut class_index = 0;

            // Insert padding to account for wordpieces.
            for (i, token) in tokens.get_tokens().iter().enumerate() {
                if tokens.get_special_tokens_mask().get(i) == Some(&1) {
                    // This is a special token, so skip it
                    continue;
                }

                // If the token is a wordpiece, we need to insert a padding token in the class_ids
                // tensor. Otherwise, we should increment the class_index.
                if token.starts_with("##") {
                    if class_index <= class_ids.len() {
                        class_ids.insert(class_index, self.batcher.pad_token_id);
                    } else {
                        // These are trailing wordpieces, so we just need to append pad tokens
                        // to the end
                        class_ids.push(self.batcher.pad_token_id);
                    }
                }

                class_index += 1;
            }

            class_ids_list.push(class_ids);
        }

        let pad_mask = generate_padding_mask(
            self.batcher.pad_token_id,
            token_ids_list,
            Some(self.batcher.max_seq_length),
            &self.batcher.device,
        );

        let seq_length = pad_mask.tensor.dims()[1];

        // Pad the slot labels to match the tokenized sequence length
        let targets = tensors::pad_to::<B>(
            self.batcher.pad_token_id,
            class_ids_list,
            seq_length,
            &self.batcher.device,
        );

        // Create and return training batch
        Train {
            input: Infer {
                tokens: pad_mask.tensor,
                mask_pad: pad_mask.mask,
            },
            targets,
        }
    }
}
