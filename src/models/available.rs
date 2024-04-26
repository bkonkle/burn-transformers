use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
};

use crate::pipelines::Pipeline;

use super::{bert::BERT_BASE_UNCASED, Model};

/// Available Models for each Pipeline
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Available(HashMap<Pipeline, Vec<Model>>);

impl Default for Available {
    fn default() -> Self {
        Self(HashMap::from([(
            Pipeline::TextClassification,
            vec![Model::Bert(BERT_BASE_UNCASED)],
        )]))
    }
}

impl Deref for Available {
    type Target = HashMap<Pipeline, Vec<Model>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Available {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
