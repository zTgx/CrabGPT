use {
    candle_core::{DType, Device, Result, Tensor},
    candle_nn::{Embedding, Module, VarBuilder, embedding},
    serde::Deserialize,
};

#[derive(Debug, Clone)]
pub struct TokenEmbeddingConfig {
    // vocabulary size
    pub vocab_size: usize,

    // The dimension of the vector after each token is embedded
    pub embedding_dim: usize,
}

impl TokenEmbeddingConfig {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            vocab_size,
            embedding_dim,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    config: TokenEmbeddingConfig,
    embedding: Embedding,
}

impl TokenEmbedding {
    pub fn new(config: TokenEmbeddingConfig, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(
            config.vocab_size,
            config.embedding_dim,
            vb.pp("token_embeddings"),
        )?;
        println!(
            "Token embedding layer shape: {:#?}",
            embedding.embeddings().shape()
        );

        Ok(Self { config, embedding })
    }
}

impl Module for TokenEmbedding {
    // Look up the embedding matrix through the token ID
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dmodel_sqrt = (self.config.embedding_dim as f32).sqrt();
        let t = Tensor::new(dmodel_sqrt, xs.device())?;

        // In the embedding layers, we multiply those weights by √dmodel.
        let token_embeddings = self.embedding.forward(xs)?.broadcast_mul(&t)?;
        println!("Multiply those weights by √dmodel: {}", token_embeddings);

        Ok(token_embeddings)
    }
}

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
    pub embedding: Tensor,
}

impl PositionEncoding {
    pub fn new(config: PositionEncodingConfig, device: &Device) -> Result<Self> {
        let embedding = PositionEncoding::pos_encoding(
            config.max_position_embeddings,
            config.embedding_dim,
            device,
        )?;
        println!("PE embedding shape: {:#?}", embedding.shape());

        Ok(Self { config, embedding })
    }

    fn pos_encoding(sequence_len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
        // 1. 生成位置张量 [sequence_len, 1]
        let position = Tensor::arange(0u32, sequence_len as u32, device)?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?; // [sequence_len, 1]
        println!("position: {}", position);

        // 2. 生成频率因子 [d_model / 2]
        let i = Tensor::arange_step(0f32, d_model as f32, 2f32, device)?.to_dtype(DType::F32)?; // [d_model / 2]

        // 3. 计算 div_term = exp(i * (-ln(10000) / d_model))
        let inv_freq = Tensor::new(-(10000f32.ln() / d_model as f32), device)?.unsqueeze(0)?; // 标量 -> [1]（广播到与i相同的形状）
        let div_term = i.mul(&inv_freq.broadcast_as(i.shape())?)?.exp()?;

        // 4. 计算位置 * div_term [sequence_len, d_model / 2]
        let pos_x_div = position.matmul(&div_term.unsqueeze(0)?)?; // [sequence_len, d_model / 2]

        // 5. 计算 sin 和 cos
        let pe_sin = pos_x_div.sin()?;
        let pe_cos = pos_x_div.cos()?;

        println!("pe_sin: {}", pe_sin);
        println!("pe_cos: {}", pe_cos);

        // 6. 交错合并 sin 和 cos [sequence_len, d_model]
        let pe = Tensor::stack(&[pe_sin, pe_cos], 2)?.flatten_from(1)?;
        Ok(pe)
    }
}

impl PositionEncoding {
    pub fn forward(&self, seq_len: usize) -> Result<Tensor> {
        // self.embedding is of shape (max_len, d_model)
        // We need to return the slice for the current sequence length.
        self.embedding.narrow(0, 0, seq_len)
    }
}

pub struct InputEmbeddings {
    pub token_embeddings: TokenEmbedding,
    pub pos_embeddings: PositionEncoding,
}

impl InputEmbeddings {
    pub fn new(token_embeddings: TokenEmbedding, pos_embeddings: PositionEncoding) -> Result<Self> {
        Ok(Self {
            token_embeddings,
            pos_embeddings,
        })
    }
}

impl InputEmbeddings {
    pub fn forward(&self, xs: &Tensor, context_len: usize) -> Result<Tensor> {
        // Positional embeddings are added to the token embedding vector to create the input embeddings for an LLM.
        // Eg. For token A
        // Token embedding: [1.0, 1.0, 1.0]
        // PE    embedding: [1.1, 1.2, 1.3]
        // Input embedding: = [1.0, 1.0, 1.0] + [1.1, 1.2, 1.3]
        // let input_embedding = token_embeddings.broadcast_add(&pos_embeddings);

        let token = self.token_embeddings.forward(xs)?;
        let pe = self.pos_embeddings.forward(context_len)?;
        token.broadcast_add(&pe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_core::Device;
    use candle_nn::VarMap;
    use tokenizers::Tokenizer;

    #[test]
    fn token_embeddings_simple_works() -> Result<()> {
        let vocab_size = 3_usize;
        let dim = 12_usize;

        let config = TokenEmbeddingConfig::new(3, 12);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let token = TokenEmbedding::new(config, vb)?;
        // println!("token embedding: {}", token.embedding.embeddings());

        assert_eq!(
            token.embedding.embeddings().shape().dims2()?,
            (vocab_size, dim)
        );

        Ok(())
    }

    #[test]
    fn token_embedding_forward_works() -> Result<()> {
        let config = TokenEmbeddingConfig::new(3, 3);

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);

        let token = TokenEmbedding::new(config, vb)?;
        println!("token embedding: {}", token.embedding.embeddings());

        let xs = Tensor::arange(0_u32, 2_u32, &Device::Cpu)?;
        println!("xs: {xs}");

        let x = token.forward(&xs)?;
        println!("x: {x}");

        Ok(())
    }

    #[test]
    fn token_embedding_works() {
        let path = "./data/tokenizer.json";
        let tokenizer = Tokenizer::from_file(path).unwrap();

        let vocab_size = tokenizer.get_vocab_size(true);

        let encoding = tokenizer.encode("Hello world", true).unwrap();
        let token_ids = encoding.get_ids();
        println!("TokenIDs: {:?}", token_ids);

        let device = Device::Cpu;
        let config = TokenEmbeddingConfig {
            vocab_size,
            embedding_dim: 512,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // New input embedding
        let input_embedding = TokenEmbedding::new(config, vb).unwrap();

        let xt = Tensor::from_slice(token_ids, (token_ids.len(),), &device).unwrap();
        let embedding = input_embedding.forward(&xt).unwrap();

        println!("Input Embedding shape: {:#?}", embedding);
    }
}

#[cfg(test)]
mod pe_tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_pos_encoding_shape() -> Result<()> {
        let device = Device::Cpu;
        let sequence_len = 768;
        let d_model = 10;

        let pe = PositionEncoding::pos_encoding(sequence_len, d_model, &device)?;
        assert_eq!(pe.shape().dims2().unwrap(), (sequence_len, d_model));

        Ok(())
    }

    #[test]
    fn test_pos_encoding() -> Result<()> {
        let device = Device::Cpu;
        let pe = PositionEncoding::pos_encoding(10, 8, &device)?;
        println!("pe: {}", pe);
        assert_eq!(pe.shape().dims(), &[10, 8]);
        Ok(())
    }

    #[test]
    fn forward_works() -> Result<()> {
        let device = Device::Cpu;
        let config = PositionEncodingConfig::default();
        let pe = PositionEncoding::new(config.clone(), &device)?;

        let xt = pe.forward(config.max_position_embeddings)?;
        assert_eq!(
            xt.shape().dims(),
            &[config.max_position_embeddings, config.embedding_dim]
        );

        Ok(())
    }

    #[test]
    fn narrow_works() {
        use candle_core::{Device, Tensor};
        let a = Tensor::new(&[[0f32, 1., 2.], [3., 4., 5.], [6., 7., 8.]], &Device::Cpu).unwrap();

        let b = a.narrow(0, 1, 2).unwrap();
        assert_eq!(b.shape().dims(), &[2, 3]);
        assert_eq!(b.to_vec2::<f32>().unwrap(), &[[3., 4., 5.], [6., 7., 8.]]);

        let c = a.narrow(1, 1, 1).unwrap();
        assert_eq!(c.shape().dims(), &[3, 1]);
        assert_eq!(c.to_vec2::<f32>().unwrap(), &[[1.], [4.], [7.]]);
    }

    #[test]
    fn stack_works() -> Result<()> {
        let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
        let b = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;

        let c = Tensor::stack(&[&a, &b], 0)?;
        assert_eq!(c.shape().dims(), &[2, 2, 3]);

        let c = Tensor::stack(&[&a, &b], 2)?;
        assert_eq!(c.shape().dims(), &[2, 3, 2]);

        Ok(())
    }
}
