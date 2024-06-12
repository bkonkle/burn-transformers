#![allow(clippy::too_many_arguments)]

use std::{collections::BTreeMap, fmt::Debug};

use burn::{
    data::dataloader,
    nn::attention::generate_padding_mask,
    tensor::{backend::Backend, Bool, Int, Tensor},
};
use derive_new::new;
use tokenizers::Tokenizer;

use crate::{pipelines::sequence_classification, utils::classes::invert_map};

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
    pub tokenizer: Tokenizer,

    /// Maximum sequence length for tokenized text
    pub max_seq_length: usize,

    /// ID of the padding token
    pub pad_token_id: usize,

    /// ID of the UNK token
    pub unk_token_id: usize,

    /// A mapping from class ids to class name labels
    pub id2label: BTreeMap<usize, String>,

    /// A mapping from class name labels to class ids
    pub label2id: BTreeMap<String, usize>,

    /// Device on which to perform computation (e.g., CPU or CUDA device)
    pub device: B::Device,
}

impl<B: Backend> Batcher<B> {
    /// Creates a new batcher
    pub fn new(
        tokenizer: Tokenizer,
        config: sequence_classification::Config,
        device: B::Device,
    ) -> Self {
        let label2id: BTreeMap<String, usize> = invert_map(config.id2label.clone());

        let unk_token_id = label2id
            .get("UNK")
            .unwrap_or(&(config.pad_token_id + 1))
            .to_owned();

        Self {
            tokenizer,
            pad_token_id: config.pad_token_id,
            unk_token_id,
            max_seq_length: config.max_seq_len.unwrap_or(config.max_position_embeddings),
            id2label: config.id2label,
            label2id,
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

        let padding = generate_padding_mask(
            self.pad_token_id,
            token_ids_list,
            Some(self.max_seq_length),
            &self.device,
        );

        // Create and return inference batch
        Infer {
            tokens: padding.tensor,
            mask_pad: padding.mask,
        }
    }
}
