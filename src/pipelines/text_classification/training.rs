use std::{collections::HashMap, path::PathBuf};

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
use tokio::{
    fs::File,
    io::{self, AsyncBufReadExt, Lines},
};

use crate::datasets::snips;

use super::{
    batcher::Train,
    pipeline::{Model, ModelConfig},
    Batcher,
};

/// Define configuration struct for the experiment
#[derive(burn::config::Config)]
pub struct Config {
    /// Maximum sequence length
    #[config(default = 50)]
    pub max_seq_length: usize,

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

    /// Model name (e.g., "bert-base-uncased")
    #[config(default = "\"bert-base-uncased\".to_string()")]
    pub model_name: String,

    /// A map from class ids to class names
    pub id2label: HashMap<usize, String>,
}

impl Config {
    /// Load configuration from file for training a particular task
    pub async fn new_for_task(task_root: &str) -> io::Result<Self> {
        // TODO: Make this generic
        let id2label = read_file(&format!("{}/intent_labels.txt", task_root))
            .await?
            .into_iter()
            .enumerate()
            .map(|(i, s)| (i, s.trim().to_string()))
            .collect::<HashMap<_, _>>();

        Ok(Config::new(id2label))
    }
}

async fn file_reader(path: &str) -> io::Result<Lines<io::BufReader<File>>> {
    let f = File::open(path).await?;

    Ok(io::BufReader::new(f).lines())
}

async fn read_file(path: &str) -> io::Result<Vec<String>> {
    let mut r = file_reader(path).await?;
    let mut lines = Vec::new();

    while let Some(line) = r.next_line().await? {
        lines.push(line);
    }

    Ok(lines)
}

/// Define train function
pub async fn train<B: AutodiffBackend, M: Model<B> + 'static, D: Dataset<snips::Item> + 'static>(
    devices: Vec<B::Device>, // Device on which to perform computation (e.g., CPU or CUDA device)
    dataset_train: D,        // Training dataset
    dataset_test: D,         // Testing dataset
    config: Config,          // Experiment configuration
    artifact_dir: &str,      // Directory to save model and config files
) -> anyhow::Result<()>
where
    i64: std::convert::From<<B as burn::tensor::backend::Backend>::IntElem>,
    M::InnerModule: ValidStep<
        Train<<B as AutodiffBackend>::InnerBackend>,
        ClassificationOutput<<B as AutodiffBackend>::InnerBackend>,
    >,
{
    let device = &devices[0];

    let (config_file, model_file) = download_hf_model(&config.model_name).await;

    let model_config = M::Config::load_pretrained(config_file, 512, config.hidden_dropout_prob)
        .map_err(|e| anyhow!("Unable to load pre-trained model config file: {}", e))?;

    let n_classes = model_config.id2label().len();

    if n_classes == 0 {
        return Err(anyhow::anyhow!(
            "Classes are not defined in the model configuration"
        ));
    }

    let model = M::load_from_safetensors(device, model_file, model_config.clone());

    // TODO: Move this logic to a trait implementation
    // Initialize the linear output
    // let output = LinearConfig::new(model_config.hidden_size(), n_classes).init(device);

    // let temp = model_config.init(device);

    // let model_record: M::Record = ModelRecord {
    //     model: from_safetensors(model_file, device, model_config),
    //     output: LinearRecord {
    //         weight: output.weight,
    //         bias: output.bias,
    //     },
    //     n_classes: ConstantRecord::new(),
    // };

    // let model = temp.load_record(model_record);

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
        .with_model_size(model_config.hidden_size())
        .init();

    // Initialize learner
    let learner = LearnerBuilder::new(artifact_dir)
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

/// Download model config and weights from Hugging Face Hub
/// If file exists in cache, it will not be downloaded again
async fn download_hf_model(model_name: &str) -> (PathBuf, PathBuf) {
    let api = hf_hub::api::tokio::Api::new().unwrap();
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
