/// Model Variants
/// --------------

/// The base model type
pub static MODEL_TYPE: &str = "bert";

/// bert-base-uncased
pub static BASE_UNCASED: &str = "bert-base-uncased";

/// bert-base-cased
pub static BASE_CASED: &str = "bert-base-cased";

/// All available BERT models
pub static ALL_MODELS: &[&str; 2] = &[BASE_UNCASED, BASE_CASED];

/// Text Classification
/// -------------------

/// Available models to use with Bert for Text Classification
pub static TEXT_CLASSIFICATION_MODELS: &[&str; 2] = &[BASE_UNCASED, BASE_CASED];

/// The default model to use
pub static DEFAULT_TEXT_CLASSIFICATION_MODEL: &str = BASE_UNCASED;
