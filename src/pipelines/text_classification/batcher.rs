#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use burn::{
    data::dataloader,
    nn::attention::generate_padding_mask,
    tensor::{backend::Backend, Bool, Data, ElementConversion, Int, Tensor},
};
use derive_new::new;
use tokenizers::Tokenizer;

use crate::datasets::snips;

use super::pipeline::ModelConfig;

/// An inference batch for text classification
#[derive(Debug, Clone, new)]
pub struct Infer<B: Backend> {
    /// Tokenized text as 2D tensor: [batch_size, max_seq_length]
    pub tokens: Tensor<B, 2, Int>,
    /// Padding mask for the tokenized text containing booleans for padding locations
    pub mask_pad: Tensor<B, 2, Bool>,
}

/// A training batch for text classification
#[derive(Clone, Debug, new)]
pub struct Train<B: Backend> {
    /// Bert Model input
    pub input: Infer<B>,

    /// Class ids for the batch
    pub targets: Tensor<B, 1, Int>,
}

/// Struct for batching sequence classification items
#[derive(Clone)]
pub struct Batcher<B: Backend> {
    /// Tokenizer for converting text to token IDs
    tokenizer: Tokenizer,

    /// Maximum sequence length for tokenized text
    max_seq_length: usize,

    /// ID of the padding token
    pad_token_id: usize,

    /// ID of the UNK token
    unk_token_id: usize,

    /// A map from class names to their corresponding ids
    class_map: HashMap<String, usize>,

    /// Device on which to perform computation (e.g., CPU or CUDA device)
    device: B::Device,
}

impl<B: Backend> Batcher<B> {
    /// Creates a new batcher
    pub fn new<C: ModelConfig>(tokenizer: Tokenizer, config: C, device: B::Device) -> Self {
        let class_map = config.label2id();

        let unk_token_id = class_map
            .get("UNK")
            .unwrap_or(&(config.pad_token_id() + 1))
            .to_owned();

        Self {
            tokenizer,
            pad_token_id: config.pad_token_id(),
            unk_token_id,
            max_seq_length: config.max_position_embeddings(),
            class_map,
            device,
        }
    }
}

/// Implement Batcher trait for Batcher struct for inference
impl<B: Backend> dataloader::batcher::Batcher<String, Infer<B>> for Batcher<B> {
    /// Collects a vector of text classification items into a inference batch
    fn batch(&self, items: Vec<String>) -> Infer<B> {
        let batch_size = items.len();

        let mut token_ids_list = Vec::with_capacity(batch_size);

        // Tokenize text and create class_id tensor for each item
        for input in items {
            let tokens = self
                .tokenizer
                .encode(input, true)
                .expect("unable to encode");

            let token_ids: Vec<_> = tokens.get_ids().iter().map(|t| *t as usize).collect();

            token_ids_list.push(token_ids);
        }

        let pad_mask = generate_padding_mask(
            self.pad_token_id,
            token_ids_list,
            Some(self.max_seq_length),
            &self.device,
        );

        // Create and return inference batch
        Infer {
            tokens: pad_mask.tensor,
            mask_pad: pad_mask.mask,
        }
    }
}

/// Implement Batcher trait for Batcher struct for training
/// TODO: Make this generic
impl<B: Backend> dataloader::batcher::Batcher<snips::Item, Train<B>> for Batcher<B> {
    /// Collects a vector of text classification items into a training batch
    fn batch(&self, items: Vec<snips::Item>) -> Train<B> {
        let batch_size = items.len();

        let infer: Infer<B> = self.batch(items.iter().map(|item| item.input.clone()).collect());

        let mut class_id_list = Vec::with_capacity(batch_size);

        // Tokenize text and create class_id tensor for each item
        for item in items {
            let class_id = self
                .class_map
                .get(&item.intent)
                .unwrap_or(&self.unk_token_id);

            class_id_list.push(Tensor::from_data(
                Data::from([(*class_id as i64).elem()]),
                &self.device,
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