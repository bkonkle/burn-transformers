#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Represents a BERT model with a specific underlying name
pub struct Model {
    /// The specific BERT-architecture model in use
    pub name: &'static str,
}

impl Model {
    /// Create a new BERT model
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }

    /// Create a new BERT model using the `bert-base-uncased` model
    pub const fn new_base_uncased() -> Self {
        Self::new(BERT_BASE_UNCASED)
    }

    /// Create a new BERT model using the `bert-base-cased` model
    pub const fn new_base_cased() -> Self {
        Self::new(BERT_BASE_CASED)
    }
}

impl crate::cli::models::Model for Model {
    fn model_type(&self) -> &str {
        BASE_MODEL
    }

    fn name(&self) -> &str {
        self.name
    }
}

/// Model Variants
/// --------------

/// The base model name
pub static BASE_MODEL: &str = "bert";

/// bert-base-uncased
pub const BERT_BASE_UNCASED: &str = "bert-base-uncased";

/// bert-base-cased
pub const BERT_BASE_CASED: &str = "bert-base-cased";
