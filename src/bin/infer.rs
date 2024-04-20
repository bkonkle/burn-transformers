//! Command line tool for training

use anyhow::{anyhow, Result};
use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};
use burn_transformers::{datasets::snips, models::bert::sequence_classification::infer};
use pico_args::Arguments;

const HELP: &str = "\
Usage: infer PIPELINE [OPTIONS]

Arguments:
  PIPELINE             The pipeline to use (e.g., 'text-classification')

Options:
  -h, --help           Print help
  -m, --model          The model to use (e.g., 'bert')
";

#[derive(Debug)]
struct Args {
    /// Prints the usage menu
    help: bool,

    /// The pipeline to use
    pipeline: String,

    /// The model to use
    model: Option<String>,
}

fn parse_args() -> Result<Args, pico_args::Error> {
    let mut pargs = Arguments::from_env();

    let args = Args {
        help: pargs.contains(["-h", "--help"]),
        pipeline: pargs.free_from_str()?,
        model: pargs.opt_value_from_str(["-m", "--model"])?,
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

    // TODO: Come up with a better mechanism for this as pipelines expand
    if let Some(model) = args.model {
        if model != "bert" {
            return Err(anyhow!("Unsupported model: {}", model));
        }
    }

    let data_root = "data";
    let task_root = format!("{}/snips", data_root);
    let artifact_dir = format!("{}/model", task_root);

    let device = LibTorchDevice::Cuda(0);

    let samples = vec![
        ("rate this novel a 3", "RateBook"),
        ("i rate shadow of suribachi at five stars", "RateBook"),
        ("play some steve boyett chant music", "PlayMusic"),
        (
            "search for the television show me and my guitar",
            "SearchCreativeWork",
        ),
        ("will it be stormy in ma", "GetWeather"),
        (
            "i d like to see movie schedules for kerasotes theatres",
            "SearchScreeningEvent",
        ),
        (
            "on jan  the twentieth what will it feel like in ct or the area not far from it",
            "GetWeather",
        ),
        ("table for 8 at a popular food court", "BookRestaurant"),
        ("add nyoil to my this is prince playlist", "AddToPlaylist"),
        (
            "find tv series titled a life in the death of joe meek",
            "SearchCreativeWork",
        ),
        (
            "show creativity of song a discord electric",
            "SearchCreativeWork",
        ),
    ];

    let input: Vec<String> = samples.iter().map(|(s, _)| (*s).to_string()).collect();

    // Get model predictions
    let (predictions, config) =
        infer::<Autodiff<LibTorch>, snips::Dataset>(device, &artifact_dir, input)?;

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
