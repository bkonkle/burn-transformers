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
};
use tokenizers::Tokenizer;

use crate::{pipelines::sequence_classification, utils::hugging_face::download_hf_model};

use super::{batcher::Train, Batcher, Item, Model, ModelConfig};

/// Training Config
pub type Config = sequence_classification::config::Training;

/// Define train function
pub async fn train<B, M, I, D>(
    devices: Vec<B::Device>, // Device on which to perform computation (e.g., CPU or CUDA device)
    dataset_train: D,        // Training dataset
    dataset_test: D,         // Testing dataset
    config: Config,          // Experiment configuration
) -> anyhow::Result<()>
where
    B: AutodiffBackend,
    M: Model<B> + 'static,
    I: Item + 'static,
    D: Dataset<I> + 'static,

    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,

    M::InnerModule: ValidStep<
        Train<<B as AutodiffBackend>::InnerBackend>,
        ClassificationOutput<<B as AutodiffBackend>::InnerBackend>,
    >,
{
    let device = &devices[0];
    let artifact_dir = format!(
        "{}/text-classification/{}",
        config.data_dir, config.model_name
    );

    let (config_file, model_file) = download_hf_model(&config.model_name).await;

    let model_config = M::Config::load_pretrained(config_file, &config.labels)
        .await
        .map_err(|e| anyhow!("Unable to load pre-trained model config file: {}", e))?;

    let model = M::load_from_safetensors(device, model_file, model_config.clone())?;

    // Initialize tokenizer
    let tokenizer = Tokenizer::from_pretrained(&config.model_name, None).unwrap();

    // Initialize batchers for training and testing data
    let batcher_train =
        Batcher::<B>::new(tokenizer.clone(), model_config.get_config(), device.clone());
    let batcher_test =
        Batcher::<B::InnerBackend>::new(tokenizer, model_config.get_config(), device.clone());

    let workers = std::thread::available_parallelism()?;

    // Initialize data loaders for training and testing data
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(workers.into())
        .build(SamplerDataset::new(dataset_train, 10_000));

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size * 2)
        .num_workers(workers.into())
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
