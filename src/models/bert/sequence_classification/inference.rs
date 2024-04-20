use burn::{
    config::Config as _,
    data::{dataloader::batcher::Batcher as BatcherTrait, dataset::Dataset},
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Tensor},
};
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::datasets::snips;

use super::{
    model::{Config, MODEL_NAME},
    Batcher,
};

/// Define inference function
pub fn infer<B: Backend, D: Dataset<snips::Item> + 'static>(
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    artifact_dir: &str, // Directory containing model and config files
    samples: Vec<String>, // Text samples for inference
) -> anyhow::Result<(Tensor<B, 2>, Config)> {
    // Load experiment configuration
    let mut config = Config::load(format!("{artifact_dir}/config.json").as_str())
        .map_err(|e| anyhow!("Unable to load config file: {}", e))?;

    config.model.hidden_dropout_prob = 0.0;

    // Initialize tokenizer
    let tokenizer = Tokenizer::from_pretrained(MODEL_NAME, None).unwrap();

    // Initialize batcher for batching samples
    let batcher = Arc::new(Batcher::<B>::new(
        tokenizer.clone(),
        &config,
        device.clone(),
    ));

    // Load pre-trained model weights
    println!("Loading weights...");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .map_err(|e| anyhow!("Unable to load trained model weights: {}", e))?;

    // Create model using loaded weights
    println!("Creating model...");

    let model = config.init(&device).load_record(record);

    // Run inference on the given text samples
    println!("Running inference...");

    let item = batcher.batch(samples.clone()); // Batch samples using the batcher

    Ok((model.infer(item), config))
}
