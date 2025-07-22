use {
    candle_core::{Result, Tensor},
    candle_nn::{Embedding, VarBuilder, embedding},
};

/// Embedding input for the model.
#[derive(Debug, Clone)]
pub struct InputEmbedding {
    pub token_embedding_layer: Embedding,
}

impl InputEmbedding {
    pub fn new(vocab_size: usize, output_dim: usize) -> Result<Self> {
        let ts = std::collections::HashMap::new();
        let dtype = candle_core::DType::F32;
        let dev = candle_core::Device::cuda_if_available(0)?;
        let vb = VarBuilder::from_tensors(ts, dtype, &dev);
        let token_embedding_layer = embedding(vocab_size, output_dim, vb)?;

        Ok(Self {
            token_embedding_layer,
        })
    }
}
