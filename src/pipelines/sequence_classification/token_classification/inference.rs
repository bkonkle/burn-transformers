use burn::{
    config::Config as _,
    data::dataloader::batcher::Batcher as BatcherTrait,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::AutodiffBackend, Tensor},
};
use std::sync::Arc;
use tokenizers::Tokenizer;

use super::{output::Inference, Batcher, Model, ModelConfig};

/// Define inference function
pub fn infer<B: AutodiffBackend, M: Model<B> + 'static>(
    device: B::Device, // Device on which to perform computation (e.g., CPU or CUDA device)
    data_dir: &str,    // The location of the top-level data directory
    model_name: &str,  // The name of the model (e.g., "bert-base-uncased")
    samples: Vec<&str>, // Text samples for inference
) -> anyhow::Result<(Vec<Tensor<B, 2>>, M::Config)>
where
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
    f64: std::convert::From<<B as burn::tensor::backend::Backend>::FloatElem>,
{
    let artifact_dir = format!("{}/token-classification/{}", data_dir, model_name);

    // Load experiment configuration
    let model_config = M::Config::load(format!("{artifact_dir}/config.json").as_str())
        .map_err(|e| anyhow!("Unable to load config file: {}", e))?;

    // Initialize tokenizer
    let tokenizer = Tokenizer::from_pretrained(model_name, None).unwrap();

    // Initialize batcher for batching samples
    let batcher = Arc::new(Batcher::<B>::new(
        tokenizer.clone(),
        model_config.get_config(),
        device.clone(),
    ));

    // Load pre-trained model weights
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .map_err(|e| anyhow!("Unable to load trained model weights: {}", e))?;

    // Create model using loaded weights
    let model = model_config.init::<B>(&device).load_record(record);

    let samples = samples.into_iter().map(|s| s.to_string()).collect();
    let item = batcher.batch(samples); // Batch samples using the batcher

    let Inference {
        output,
        attention_mask,
    } = model.infer(item);

    let [batch_size, seq_length, n_classes] = output.dims();

    let mut predictions = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let indices = attention_mask
            .clone()
            .slice([b..b + 1, 0..seq_length])
            .reshape([seq_length])
            .nonzero()
            .remove(0);

        let prediction = output
            .clone()
            .slice([b..b + 1, 0..seq_length, 0..n_classes])
            .reshape([seq_length, n_classes])
            .select(0, indices);

        predictions.push(prediction);
    }

    // Run inference on the given text samples, and return the config for reference
    Ok((predictions, model_config.clone()))
}
