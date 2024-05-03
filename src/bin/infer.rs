//! Command line tool for training

use anyhow::{anyhow, Result};
use burn::{
    backend::{libtorch::LibTorchDevice, Autodiff, LibTorch},
    data::dataloader::Dataset as _,
};
use burn_transformers::{
    datasets::{snips, Dataset, LoadableDataset},
    models::bert::sequence_classification,
    pipelines::text_classification::{self, infer},
};
use pico_args::Arguments;
use rand::Rng;

const HELP: &str = "\
Usage: infer PIPELINE DATASET [OPTIONS]

Arguments:
  PIPELINE             The pipeline to use (e.g., 'text-classification')
  DATASET              The dataset to use (e.g., 'snips')

Options:
  -h, --help           Print help
  -m, --model          The model to use (e.g., 'bert-base-uncased')
  -d, --data-dir       The path to the top-level data directory (defaults to 'data')
";

#[derive(Debug)]
struct Args {
    pipeline: String,
    dataset: String,
    model: Option<String>,
    data_dir: Option<String>,
}

impl Args {
    fn parse() -> anyhow::Result<Option<Self>> {
        let mut pargs = Arguments::from_env();

        // Help has a higher priority and should be handled separately.
        if pargs.contains(["-h", "--help"]) {
            return Ok(None);
        }

        let args = Args {
            pipeline: pargs.free_from_str()?,
            model: pargs.opt_value_from_str(["-m", "--model"])?,
            data_dir: pargs.opt_value_from_str(["-d", "--data-dir"])?,
            dataset: pargs.free_from_str().map_err(|e| match e {
                pico_args::Error::MissingArgument => anyhow!("Missing required argument: DATASET"),
                _ => anyhow!("{}", e),
            })?,
        };

        Ok(Some(args))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let output = Args::parse()?;

    if output.is_none() {
        print!("{}", HELP);

        return Ok(());
    }
    let args = output.unwrap();

    // TODO: Come up with a better mechanism for this as pipelines expand
    if args.pipeline != "text-classification" {
        return Err(anyhow!("Unsupported pipeline: {}", args.pipeline));
    }

    let data_dir = args.data_dir.unwrap_or_else(|| "data".to_string());

    // TODO: Come up with a better mechanism for this as pipelines expand
    let model = args
        .model
        .map_or(Ok("bert-base-uncased".to_string()), |model| {
            if model != "bert-base-uncased" {
                return Err(anyhow!("Unsupported model: {}", model));
            }

            Ok(model)
        })?;

    let device = LibTorchDevice::Cuda(0);

    // TODO: Use this value to dynamically load the dataset to train with
    let dataset = Dataset::try_from(args.dataset)?;
    let data: snips::Dataset = LoadableDataset::load(&data_dir, "test").await?;

    // let data = snips::Dataset::load(&data_dir, "test").await?;

    let mut rng = rand::thread_rng();

    let mut samples = Vec::with_capacity(10);
    for _ in 0..10 {
        let i = rng.gen_range(0..data.len());
        let item = data.get(i).unwrap();

        samples.push((item.input.clone(), item.intent.clone()));
    }

    let input: Vec<String> = samples.iter().map(|(s, _)| (*s).clone()).collect();

    // Get model predictions
    // TODO: Make this more generic
    let (predictions, config) = infer::<
        Autodiff<LibTorch>,
        sequence_classification::Model<Autodiff<LibTorch>>,
    >(device, data_dir, &model, input)?;

    // Print out predictions for each sample
    for (i, (text, expected)) in samples.into_iter().enumerate() {
        // Get predictions for current sample
        #[allow(clippy::single_range_in_vec_init)]
        let prediction = predictions.clone().slice([i..i + 1]);

        let class_indexes = prediction.argmax(1).into_data().convert::<i64>().value;

        let classes = class_indexes
            .into_iter()
            .map(|index| &config.id2label[&(index as usize)])
            .collect::<Vec<_>>();

        let class = classes.first().unwrap();

        // Print sample text and predicted class name
        println!(
            "\n=== Item {i} ===\
             \n- Text: {text}\
             \n- Class: {class}\
             \n- Expected: {expected}\
             \n================"
        );
    }

    Ok(())
}
