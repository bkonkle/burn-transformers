///  The Snips dataset
pub mod snips;

use async_trait::async_trait;

/// A datates which can be loaded
#[async_trait]
pub trait LoadableDataset<I>: burn::data::dataset::Dataset<I> {
    /// Load the dataset
    async fn load(data_dir: &str, mode: &str) -> std::io::Result<Self>
    where
        Self: std::marker::Sized;
}
