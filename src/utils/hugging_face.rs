use std::path::PathBuf;

use hf_hub::api::tokio;

/// Download model config and weights from Hugging Face Hub
/// If file exists in cache, it will not be downloaded again
// NOTE: Modified from the built-in function to work within an already-async context
pub async fn download_hf_model(model_name: &str) -> (PathBuf, PathBuf) {
    let api = tokio::Api::new().unwrap();
    let repo = api.model(model_name.to_string());

    let model_filepath = repo.get("model.safetensors").await.unwrap_or_else(|_| {
        panic!(
            "Failed to download: {} weights with name: model.safetensors from HuggingFace Hub",
            model_name
        )
    });

    let config_filepath = repo.get("config.json").await.unwrap_or_else(|_| {
        panic!(
            "Failed to download: {} config with name: config.json from HuggingFace Hub",
            model_name
        )
    });

    (config_filepath, model_filepath)
}
