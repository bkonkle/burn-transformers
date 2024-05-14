use burn::tensor::{backend::Backend, Data, ElementConversion, Int, Shape, Tensor};

/// Generation padding to a specific max length, typically to correlate with tokenzed sequences
pub fn pad_to<B: Backend>(
    pad_token: usize,
    tokens_list: Vec<Vec<usize>>,
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch_size = tokens_list.len();

    let mut tensor = Tensor::zeros([batch_size, seq_length], device);
    tensor = tensor.add_scalar(pad_token as i64);

    for (index, tokens) in tokens_list.into_iter().enumerate() {
        let seq_length = tokens.len();

        tensor = tensor.slice_assign(
            [index..index + 1, 0..tokens.len()],
            Tensor::from_data(
                Data::new(
                    tokens.into_iter().map(|e| (e as i64).elem()).collect(),
                    Shape::new([1, seq_length]),
                ),
                device,
            ),
        );
    }

    tensor
}
