/// BERT for Text Classification Configuration
pub mod config;

/// BERT for Text Classification
pub mod model;

/// Training routine
pub mod train;

pub use config::Config;
pub use model::{Model, ModelRecord};
