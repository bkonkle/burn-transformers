//! Command line tool to trigger training (WIP)

use anyhow::{anyhow, Result};
use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};
use burn_transformers::{
    datasets::snips,
    models::{self, bert::sequence_classification},
    pipelines::text_classification::training::{self, Config},
};
use pico_args::Arguments;

const HELP: &str = "\
Usage: train PIPELINE DATASET [OPTIONS]

Arguments:
  PIPELINE             The pipeline to use (e.g., 'text-classification')
  DATASET              The dataset to use (e.g., 'snips')

Options:
  -h, --help           Print help
  -m, --model          The model to use (e.g., 'bert')
  -n, --num-epochs     Number of epochs to train for
  -b, --batch-size     Batch size
";

#[derive(Debug)]
struct Args {
    /// Prints the usage menu
    help: bool,

    /// The pipeline to use
    pipeline: String,

    /// The dataset to use
    dataset: String,

    /// The model to use
    model: Option<String>,

    /// Number of epochs to train for
    num_epochs: Option<usize>,

    /// Batch size
    batch_size: Option<usize>,
}

fn parse_args() -> Result<Args, pico_args::Error> {
    let mut pargs = Arguments::from_env();

    let free: std::path::PathBuf = pargs.free_from_str()?;

    // Split the free text on whitespace
    let tokens: Vec<_> = free
        .to_str()
        .unwrap_or_default()
        .split_whitespace()
        .collect();

    let args = Args {
        help: pargs.contains(["-h", "--help"]),
        pipeline: tokens[0].to_string(),
        dataset: tokens[1].to_string(),
        model: pargs.opt_value_from_str(["-m", "--model"])?,
        num_epochs: pargs.opt_value_from_str(["-n", "--num-epochs"])?,
        batch_size: pargs.opt_value_from_str(["-b", "--batch-size"])?,
    };

    Ok(args)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;

    if args.help {
        println!("{}", HELP);
        return Ok(());
    }

    // TODO: Come up with a better mechanism for this as pipelines expand
    if args.pipeline != "text-classification" {
        return Err(anyhow!("Unsupported pipeline: {}", args.pipeline));
    }

    // TODO: Come up with a better mechanism for this as datasets expand
    if args.dataset != "snips" {
        return Err(anyhow!("Unsupported pipeline: {}", args.pipeline));
    }

    // TODO: Come up with a better mechanism for this as models expand
    let model = args.model.map_or(Ok("bert".to_string()), |model| {
        if model != "bert" {
            return Err(anyhow!("Unsupported model: {}", model));
        }

        Ok(model)
    })?;

    let dataset_root = format!("data/datasets/{}", args.dataset);
    let artifact_dir = format!("data/pipelines/{}/{}", args.pipeline, model);

    let mut config = Config::new_for_task(&dataset_root).await?;

    if let Some(num_epochs) = args.num_epochs {
        config.num_epochs = num_epochs;
    }

    if let Some(batch_size) = args.batch_size {
        config.batch_size = batch_size;
    }

    let device = LibTorchDevice::Cuda(0);

    // TODO: Make this more generic
    training::train::<
        Autodiff<LibTorch>,
        sequence_classification::Model<Autodiff<LibTorch>>,
        snips::Dataset,
    >(
        vec![device],
        snips::Dataset::train().await?,
        snips::Dataset::test().await?,
        config,
        &artifact_dir,
    )
    .await?;

    Ok(())
}
