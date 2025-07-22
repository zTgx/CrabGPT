use {
    candle_core::{Result, Tensor},
    candle_nn::{Module, VarBuilder, embedding},
    serde::Deserialize,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,

    Relative,
}

pub struct PositionEncodingConfig {
    pub max_position_embeddings: usize,
    pub embedding_dim: usize,
    pub position_embedding_type: PositionEmbeddingType,
}

impl PositionEncodingConfig {
    pub fn new(
        max_position_embeddings: usize,
        embedding_dim: usize,
        position_embedding_type: PositionEmbeddingType,
    ) -> Self {
        Self {
            max_position_embeddings,
            embedding_dim,
            position_embedding_type,
        }
    }
}

impl Default for PositionEncodingConfig {
    fn default() -> Self {
        Self {
            max_position_embeddings: 512,
            embedding_dim: 768,
            position_embedding_type: PositionEmbeddingType::Absolute,
        }
    }
}

pub struct PositionEncoding {
    pub config: PositionEncodingConfig,
    pub pos_embedding: Tensor,
}

impl PositionEncoding {
    pub fn new(config: PositionEncodingConfig) -> Result<Self> {
        let ts = std::collections::HashMap::new();
        let dtype = candle_core::DType::F32;
        let dev = candle_core::Device::cuda_if_available(0)?;
        let vb = VarBuilder::from_tensors(ts, dtype, &dev);
        let pos_embedding_layer =
            embedding(config.max_position_embeddings, config.embedding_dim, vb)?;

        let xs = Tensor::arange(0, config.max_position_embeddings as i64, &dev)?;
        let xs = xs.reshape((1, config.max_position_embeddings))?;
        let pos_embedding = pos_embedding_layer.forward(&xs)?;

        Ok(Self {
            config,
            pos_embedding,
        })
    }
}
