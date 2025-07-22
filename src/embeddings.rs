use {
    crate::pe::{PositionEncoding, PositionEncodingConfig},
    candle_core::{Result, Tensor},
    candle_nn::{Embedding, Module, VarBuilder, embedding},
};

pub struct InputEmbeddingConfig {
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub max_position_embeddings: usize,
}

/// Embedding input for the model.
#[derive(Debug, Clone)]
pub struct InputEmbedding {
    token_embedding_layer: Embedding,
    pos_encoding_layer: PositionEncoding,
}

impl InputEmbedding {
    pub fn new(config: InputEmbeddingConfig, vb: VarBuilder) -> Result<Self> {
        let token_embedding_layer = embedding(
            config.vocab_size,
            config.embedding_dim,
            vb.pp("token_embeddings"),
        )?;

        let pos_config = PositionEncodingConfig::new(
            config.max_position_embeddings,
            config.embedding_dim,
            crate::pe::PositionEmbeddingType::Absolute,
        );
        let pos_encoding_layer = PositionEncoding::new(pos_config, vb.device())?;

        Ok(Self {
            token_embedding_layer,
            pos_encoding_layer,
        })
    }
}

impl Module for InputEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Get the sequence length from the input tensor shape
        let (_batch_size, seq_len) = xs.dims2()?;

        // Get token embeddings and positional embeddings
        let token_embeddings = self.token_embedding_layer.forward(xs)?;
        let pos_embeddings = self.pos_encoding_layer.forward(seq_len)?;

        // Add them together
        token_embeddings.add(&pos_embeddings)
    }
}
