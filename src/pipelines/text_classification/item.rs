use std::fmt::Debug;

/// A trait for items that can be used for text classification
pub trait Item: Send + Sync + Clone + Debug {
    /// Returns the input text for the item
    fn input(&self) -> &str;

    /// Returns the class label for the item
    fn class_label(&self) -> &str;
}
