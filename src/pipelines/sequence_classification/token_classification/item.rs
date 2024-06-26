use std::fmt::Debug;

/// A trait for items that can be used for text classification
pub trait Item: Send + Sync + Clone + Debug {
    /// Returns the input text for the item
    fn input(&self) -> &str;

    /// Returns the token class labels for the item
    fn class_labels(&self) -> Vec<&str>;
}
