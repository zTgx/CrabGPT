use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, ModuleT, VarBuilder, VarMap};
use crabgpt::{
    embeddings::{InputEmbedding, InputEmbeddingConfig},
    transformer::{DecoderBlock, TransformerBlock},
};

fn main() -> Result<()> {
    // 1. 初始化设备和参数
    let device = Device::Cpu;
    let vocab_size = 10000;
    let embedding_dim = 768;
    let max_position_embeddings = 512;
    let num_heads = 12;
    let context_length = 32;
    let drop_p = 0.1;

    // 2. 构建参数管理器
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // 3. 构建输入嵌入层
    let embed_config = InputEmbeddingConfig {
        vocab_size,
        embedding_dim,
        max_position_embeddings,
    };
    let src_embedding = InputEmbedding::new(embed_config.clone(), vb.pp("src_embeddings"))?;
    let tgt_embedding = InputEmbedding::new(embed_config, vb.pp("tgt_embeddings"))?;

    // 4. 构建 Encoder 和 Decoder 块
    let encoder = TransformerBlock::new(
        embedding_dim,
        num_heads,
        context_length,
        drop_p,
        vb.pp("encoder"),
    )?;
    let decoder = DecoderBlock::new(
        embedding_dim,
        num_heads,
        context_length,
        drop_p,
        vb.pp("decoder"),
    )?;

    // 5. 构造输入（batch_size=2, seq_len=32）
    let src_ids =
        Tensor::arange(0, (2 * context_length) as u32, &device)?.reshape((2, context_length))?;
    let tgt_ids =
        Tensor::arange(0, (2 * context_length) as u32, &device)?.reshape((2, context_length))?;

    // 6. 编码器前向
    let src_embedded = src_embedding.forward(&src_ids)?;
    let encoder_output = encoder.forward_t(&src_embedded, true)?;

    // 7. 解码器前向（带 encoder 输出）
    let tgt_embedded = tgt_embedding.forward(&tgt_ids)?;
    let decoder_output = decoder.forward_t(&tgt_embedded, &encoder_output, true)?;

    // [2, 32, 768]
    println!("Encoder output shape: {:?}", encoder_output.shape());

    // [2, 32, 768]
    println!("Decoder output shape: {:?}", decoder_output.shape());
    Ok(())
}
