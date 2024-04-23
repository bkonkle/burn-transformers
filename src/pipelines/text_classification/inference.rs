use burn::{
    config::Config as _,
    data::{dataloader::batcher::Batcher as BatcherTrait, dataset::Dataset},
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::AutodiffBackend, Tensor},
};
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::datasets::snips;

use super::{
    Batcher, {Model, ModelConfig},
};

/// Define inference function
pub fn infer<B: AutodiffBackend, M: Model<B> + 'static, D: Dataset<snips::Item> + 'static>(
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    model_name: &str,  // The name of the model (e.g., "bert-base-uncased")
    artifact_dir: &str, // Directory containing model and config files
    samples: Vec<String>, // Text samples for inference
) -> anyhow::Result<(Tensor<B, 2>, M::Config)>
where
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    // Load experiment configuration
    let config = M::Config::load(format!("{artifact_dir}/config.json").as_str())
        .map_err(|e| anyhow!("Unable to load config file: {}", e))?;

    // Initialize tokenizer
    let tokenizer = Tokenizer::from_pretrained(model_name, None).unwrap();

    // Initialize batcher for batching samples
    let batcher = Arc::new(Batcher::<B>::new(
        tokenizer.clone(),
        config.clone(),
        device.clone(),
    ));

    // Load pre-trained model weights
    println!("Loading weights...");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .map_err(|e| anyhow!("Unable to load trained model weights: {}", e))?;

    // Create model using loaded weights
    println!("Creating model...");

    let model = config.init::<B>(&device).load_record(record);

    // Run inference on the given text samples
    println!("Running inference...");

    let item = batcher.batch(samples.clone()); // Batch samples using the batcher

    Ok((model.infer(item), config.clone()))
}
