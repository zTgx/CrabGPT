use {
    candle_core::{DType, Device, Result, Tensor},
    serde::Deserialize,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,

    Relative,
}

#[derive(Debug, Clone)]
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

/// PE = Position Encoding
/// Attention is all your need: https://arxiv.org/pdf/1706.03762
/// Positional Encoding: In order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.
/// Chose this function BECAUSE we hypothesiszed it would allow the model to easily learn to attend by relative position.
/// - For each position, the even rows are given: 2i     -> PE(pos,2i)   = sin(pos/10000^(2i/d_model))
/// - For each position, the odd  rows are given: 2i + 1 -> PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
/// - pos     : token 在序列中的位置
/// - i       : 维度索引
/// - d_model : 模型的 embedding dim

#[derive(Debug, Clone)]
pub struct PositionEncoding {
    pub config: PositionEncodingConfig,
    pub pos_embedding: Tensor,
}

impl PositionEncoding {
    pub fn new(config: PositionEncodingConfig, device: &Device) -> Result<Self> {
        let pos_embedding = PositionEncoding::pos_encoding(
            config.max_position_embeddings,
            config.embedding_dim,
            device,
        )?;

        Ok(Self {
            config,
            pos_embedding,
        })
    }

    pub fn forward(&self, seq_len: usize) -> Result<Tensor> {
        // self.pos_embedding is of shape (max_len, d_model)
        // We need to return the slice for the current sequence length.
        self.pos_embedding.narrow(0, 0, seq_len)
    }

    fn pos_encoding(sequence_len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
        // 在维度 1 上增加一个新维度，将其形状变为 (sequence_len, 1), 现在，它变成了一个列向量（一个只有一列的矩阵）.
        let position = Tensor::arange(0u32, sequence_len as u32, device)?
            .to_dtype(DType::F32)? //eg. shape: (4) -> [0., 1.0, 2.0, 3.0]
            .unsqueeze(1)?; //eg. shape: (4,1) -> [[0.], 
        //  [1.0],
        //  [2.0],
        //  [3.0]]

        // Calculate div_term directly on the device.
        let i = Tensor::arange_step(0f32, d_model as f32, 2f32, device)?;
        let inv_freq = Tensor::new(-(10000f32.ln() / d_model as f32), device)?;
        let div_term = (i * inv_freq)?.exp()?;

        // 在维度 0 上增加一个新维度，将其形状变为 (1, d_model / 2), 现在，它变成了一个行向量（一个只有一行的矩阵）。
        // 乘法 (sequence_len, 1) @ (1, d_model / 2) 的结果是一个形状为 (sequence_len, d_model / 2).
        let pos_x_div = position.matmul(&div_term.unsqueeze(0)?)?;
        let pe_sin = pos_x_div.sin()?;
        let pe_cos = pos_x_div.cos()?;

        // Interleave sin and cos in a more idiomatic way.
        // Stack to (seq_len, d_model/2, 2) and then flatten to (seq_len, d_model).
        let pe = Tensor::stack(&[pe_sin, pe_cos], 2)?.flatten_from(1)?;
        Ok(pe)
    }
}
