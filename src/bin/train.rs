//! Command line tool to trigger training (WIP)

use anyhow::anyhow;
use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};
use burn_transformers::{
    datasets::{snips, Dataset},
    models::{bert::sequence_classification, Model},
    pipelines::{
        text_classification::training::{self, Config},
        Pipeline,
    },
};
use pico_args::Arguments;

const HELP: &str = "\
Usage: train PIPELINE DATASET [OPTIONS]

Arguments:
  PIPELINE             The pipeline to use (e.g., 'text-classification')
  DATASET              The dataset to use (e.g., 'snips')

Options:
  -h, --help           Print help
  -m, --model          The model to use (e.g., 'bert-base-uncased')
  -d, --data-dir       The path to the top-level data directory (defaults to 'data')
  -n, --num-epochs     Number of epochs to train for
  -b, --batch-size     Batch size
  -d, --data-dir       The path to the top-level data directory (defaults to 'data')
";

#[derive(Debug)]
struct Args {
    pipeline: String,
    dataset: String,
    model: Option<String>,
    num_epochs: Option<usize>,
    batch_size: Option<usize>,
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
            model: pargs.opt_value_from_str(["-m", "--model"])?,
            num_epochs: pargs.opt_value_from_str(["-n", "--num-epochs"])?,
            batch_size: pargs.opt_value_from_str(["-b", "--batch-size"])?,
            data_dir: pargs.opt_value_from_str(["-d", "--data-dir"])?,
            pipeline: pargs.free_from_str().map_err(|e| match e {
                pico_args::Error::MissingArgument => anyhow!("Missing required argument: PIPELINE"),
                _ => anyhow!("{}", e),
            })?,
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

    let pipeline = Pipeline::try_from(args.pipeline)?;

    // TODO: Use this value to dynamically load the dataset to train with
    let dataset = Dataset::try_from(args.dataset)?;

    let model = if let Some(model) = args.model.clone() {
        Model::try_from(model)?
    } else {
        pipeline.default_model()
    };

    let mut config = Config::new(model.into(), dataset.into());

    if let Some(num_epochs) = args.num_epochs {
        config.num_epochs = num_epochs;
    }

    if let Some(batch_size) = args.batch_size {
        config.batch_size = batch_size;
    }

    if let Some(data_dir) = args.data_dir {
        config.data_dir = data_dir;
    }

    let device = LibTorchDevice::Cuda(0);

    match pipeline {
        Pipeline::TextClassification => {
            training::train::<
                Autodiff<LibTorch>,
                sequence_classification::Model<Autodiff<LibTorch>>,
                snips::Item,
                snips::Dataset,
            >(
                vec![device],
                snips::Dataset::load(&config.data_dir, "train").await?,
                snips::Dataset::load(&config.data_dir, "test").await?,
                config,
            )
            .await?;
        }
    }

    Ok(())
}
