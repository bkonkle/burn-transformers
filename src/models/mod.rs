// /// Available models for each pipeline1
// pub mod available;

/// BERT variants
pub mod bert;

// pub use available::Available;

// /// Available Models
// #[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
// pub enum Model {
//     /// BERT / RoBERTa
//     Bert(&'static str),
// }

// impl TryFrom<String> for Model {
//     type Error = ModelError;

//     fn try_from(value: String) -> Result<Self, Self::Error> {
//         for &model in bert::MODELS.iter() {
//             if *model == value {
//                 return Ok(Self::Bert(model));
//             }
//         }

//         Err(ModelError::Unknown(value))
//     }
// }

// impl From<Model> for String {
//     fn from(model: Model) -> Self {
//         match model {
//             Model::Bert(model) => model.to_string(),
//         }
//     }
// }

// /// Model Error
// #[derive(thiserror::Error, Debug)]
// pub enum ModelError {
//     /// No model found for the given string
//     #[error("no model found for {0}")]
//     Unknown(String),
// }
