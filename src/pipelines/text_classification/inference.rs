use burn::{
    config::Config as _,
    data::dataloader::batcher::Batcher as BatcherTrait,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::AutodiffBackend, Tensor},
};
use std::sync::Arc;
use tokenizers::Tokenizer;

use super::{Batcher, Model, ModelConfig};

/// Define inference function
pub fn infer<B: AutodiffBackend, M: Model<B> + 'static>(
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    data_dir: &str,    // The location of the top-level data directory
    model_name: &str,  // The name of the model (e.g., "bert-base-uncased")
    samples: Vec<&str>, // Text samples for inference
) -> anyhow::Result<(Tensor<B, 2>, M::Config)>
where
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
{
    let artifact_dir = format!("{}/pipelines/text-classification/{}", data_dir, model_name);

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
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .map_err(|e| anyhow!("Unable to load trained model weights: {}", e))?;

    // Create model using loaded weights
    let model = config.init::<B>(&device).load_record(record);

    let samples = samples.into_iter().map(|s| s.to_string()).collect();
    let item = batcher.batch(samples); // Batch samples using the batcher

    // Run inference on the given text samples, and return the config for reference
    Ok((model.infer(item), config.clone()))
}
