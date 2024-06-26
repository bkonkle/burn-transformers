/// BERT for Token Classification Config
pub mod config;

/// BERT for Token Classification
pub mod model;

/// Training routine
pub mod train;

pub use config::Config;
pub use model::{Model, ModelRecord};
