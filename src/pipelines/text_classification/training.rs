use burn::{
    config::Config as _,
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{transform::SamplerDataset, Dataset},
    },
    lr_scheduler::noam::NoamLrSchedulerConfig,
    optim::AdamWConfig,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric},
        ClassificationOutput, LearnerBuilder, ValidStep,
    },
    LearningRate,
};
use tokenizers::Tokenizer;

use crate::{datasets::snips, utils::hugging_face::download_hf_model};

use super::{
    batcher::Train,
    Batcher, {Model, ModelConfig},
};

/// Define configuration struct for the experiment
#[derive(burn::config::Config)]
pub struct Config {
    /// Batch size
    #[config(default = 2)]
    pub batch_size: usize,

    /// Number of epochs
    #[config(default = 1)]
    pub num_epochs: usize,

    /// Adam epsilon
    #[config(default = 1e-8)]
    pub adam_epsilon: f32,

    /// Initial learning rate
    #[config(default = 5e-5)]
    pub learning_rate: LearningRate,

    /// Dropout rate
    #[config(default = 0.1)]
    pub hidden_dropout_prob: f64,

    /// The location of the top-level data directory
    #[config(default = "\"data\".to_string()")]
    pub data_dir: String,

    /// Model name (e.g., "bert-base-uncased")
    pub model_name: String,

    /// The Dataset to use (e.g., "snips")
    pub dataset_name: String,
}

/// Define train function
pub async fn train<B, M, D>(
    devices: Vec<B::Device>, // Device on which to perform computation (e.g., CPU or CUDA device)
    dataset_train: D,        // Training dataset
    dataset_test: D,         // Testing dataset
    config: Config,          // Experiment configuration
) -> anyhow::Result<()>
where
    B: AutodiffBackend,
    M: Model<B> + 'static,
    D: Dataset<snips::Item> + 'static,

    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,

    M::InnerModule: ValidStep<
        Train<<B as AutodiffBackend>::InnerBackend>,
        ClassificationOutput<<B as AutodiffBackend>::InnerBackend>,
    >,
{
    let device = &devices[0];
    let dataset_dir = format!("{}/datasets/{}", config.data_dir, config.dataset_name);
    let artifact_dir = format!(
        "{}/pipelines/text-classification/{}",
        config.data_dir, config.model_name
    );

    let (config_file, model_file) = download_hf_model(&config.model_name).await;

    let model_config = M::Config::load_pretrained(config_file, &dataset_dir)
        .await
        .map_err(|e| anyhow!("Unable to load pre-trained model config file: {}", e))?;

    let model = M::load_from_safetensors(device, model_file, model_config.clone())?;

    // Initialize tokenizer
    let tokenizer = Tokenizer::from_pretrained(&config.model_name, None).unwrap();

    // Initialize batchers for training and testing data
    let batcher_train = Batcher::<B>::new(tokenizer.clone(), model_config.clone(), device.clone());
    let batcher_test =
        Batcher::<B::InnerBackend>::new(tokenizer, model_config.clone(), device.clone());

    // Initialize data loaders for training and testing data
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_train, 10_000));

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size * 2)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_test, 1_000));

    // Initialize optimizer
    let optimizer = AdamWConfig::new().with_epsilon(config.adam_epsilon).init();

    // Initialize learning rate scheduler
    let lr_scheduler = NoamLrSchedulerConfig::new(config.learning_rate)
        .with_warmup_steps(0)
        .with_model_size(model_config.get_config().hidden_size)
        .init();

    // Initialize learner
    let learner = LearnerBuilder::new(&artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(devices)
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, optimizer, lr_scheduler);

    // Train the model
    let model_trained = learner.fit(dataloader_train, dataloader_test);

    // Save the configuration and the trained model
    model_config
        .save(format!("{artifact_dir}/config.json"))
        .unwrap();

    CompactRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();

    Ok(())
}
