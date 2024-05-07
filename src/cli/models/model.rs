/// A Model trait for taxonomy, used in the CLI options
pub trait Model {
    /// The name of the model
    fn name(&self) -> &str;

    /// The type of model
    fn model_type(&self) -> &str;
}
