//! Command line tool for training

use anyhow::anyhow;
use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};
use burn_transformers::{
    cli::{datasets::Dataset, models::Model, pipelines::Pipeline},
    datasets::snips,
    models::bert,
    pipelines::sequence_classification,
};
use pico_args::Arguments;

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
async fn main() -> anyhow::Result<()> {
    let output = Args::parse()?;

    if output.is_none() {
        print!("{}", HELP);

        return Ok(());
    }
    let args = output.unwrap();

    let pipeline = Pipeline::try_from(args.pipeline.as_str())?;

    let model = if let Some(model) = args.model.clone() {
        Model::try_from(model.as_str())?
    } else {
        pipeline.default_model()
    };

    if !model.is_supported(&pipeline) {
        return Err(anyhow!(
            "Model \"{}\" is not supported by pipeline \"{}\"",
            model,
            pipeline
        ));
    }

    let data_dir = args.data_dir.unwrap_or_else(|| "data".to_string());

    let dataset = Dataset::try_from(args.dataset.as_str())?;

    match pipeline {
        Pipeline::TextClassification => {
            handle_text_classification(&model, &dataset, &data_dir).await
        }
        Pipeline::TokenClassification => {
            handle_token_classification(&model, &dataset, &data_dir).await
        }
    }
}

async fn handle_text_classification(
    model: &Model,
    dataset: &Dataset,
    data_dir: &str,
) -> anyhow::Result<()> {
    let samples = match model {
        Model::Bert(_) => match dataset {
            Dataset::Snips => snips::Dataset::get_samples("test").await?,
        },
    };

    let input: Vec<_> = samples.iter().map(|(s, _, _)| s.as_str()).collect();

    let device = LibTorchDevice::Cuda(0);

    // Get model predictions
    let (predictions, config) = sequence_classification::text_classification::infer::<
        Autodiff<LibTorch>,
        bert::text_classification::Model<Autodiff<LibTorch>>,
    >(device, data_dir, &model.to_string(), input)?;

    // Print out predictions for each sample
    for (i, (text, expected, _)) in samples.into_iter().enumerate() {
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

async fn handle_token_classification(
    model: &Model,
    dataset: &Dataset,
    data_dir: &str,
) -> anyhow::Result<()> {
    let samples = match model {
        Model::Bert(_) => match dataset {
            Dataset::Snips => snips::Dataset::get_samples("test").await?,
        },
    };

    let input: Vec<_> = samples.iter().map(|(s, _, _)| s.as_str()).collect();

    let device = LibTorchDevice::Cuda(0);

    // Get model predictions
    let (predictions, config) = sequence_classification::token_classification::infer::<
        Autodiff<LibTorch>,
        bert::token_classification::Model<Autodiff<LibTorch>>,
    >(device, data_dir, &model.to_string(), input)?;

    // Print out predictions for each sample
    for (i, (text, _, expected)) in samples.into_iter().enumerate() {
        // Get predictions for current sample
        #[allow(clippy::single_range_in_vec_init)]
        let prediction = &predictions[i];

        let class_indexes = prediction
            .clone()
            .argmax(1)
            .into_data()
            .convert::<i64>()
            .value;

        let classes = class_indexes
            .into_iter()
            .map(|index| config.id2label[&(index as usize)].clone())
            .collect::<Vec<_>>();

        let tokens = classes.join(" ");

        // Print sample text and predicted class name
        println!(
            "\n=== Item {i} ===\
             \n- Text: {text}\
             \n- Tokens:   {tokens}\
             \n- Expected: {expected}\
             \n================"
        );
    }

    Ok(())
}
